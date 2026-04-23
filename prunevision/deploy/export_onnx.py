"""
PruneVision AI - ONNX Export
Export pruned models to ONNX format for cross-platform deployment.
"""

import os
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def export_to_onnx(
    model: nn.Module,
    output_path: Optional[str] = None,
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    opset_version: int = None,
    dynamic_axes: Optional[Dict] = None,
) -> str:
    """
    Export a PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export.
        output_path: Path to save the ONNX model.
        input_size: Shape of dummy input tensor.
        opset_version: ONNX opset version.
        dynamic_axes: Dynamic axes specification.
        
    Returns:
        Path to saved ONNX model.
    """
    opset_version = opset_version or config.ONNX_OPSET_VERSION
    dynamic_axes = dynamic_axes or config.ONNX_DYNAMIC_AXES
    
    if output_path is None:
        output_path = os.path.join(config.EXPORT_DIR, "prunevision_model.onnx")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model.eval()
    dummy_input = torch.randn(input_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    
    # Get file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[ONNX Export] Model saved to {output_path} ({size_mb:.2f} MB)")
    
    return output_path


def validate_onnx_model(
    onnx_path: str,
    pytorch_model: Optional[nn.Module] = None,
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    tolerance: float = 1e-4,
) -> Dict:
    """
    Validate an ONNX model: check structure and optionally compare
    outputs with PyTorch model.
    
    Args:
        onnx_path: Path to ONNX model.
        pytorch_model: Optional PyTorch model for output comparison.
        input_size: Input tensor shape.
        tolerance: Max allowed difference between PyTorch and ONNX outputs.
        
    Returns:
        Validation report dict.
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        return {
            "valid": False,
            "error": "onnx and onnxruntime packages required. Install with: pip install onnx onnxruntime",
        }
    
    report = {"valid": True, "errors": []}
    
    # Check ONNX model structure
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        report["model_valid"] = True
    except Exception as e:
        report["valid"] = False
        report["errors"].append(f"ONNX validation failed: {str(e)}")
        return report
    
    # Run inference with ONNX Runtime
    try:
        session = ort.InferenceSession(onnx_path)
        dummy_input = torch.randn(input_size).numpy()
        
        ort_inputs = {session.get_inputs()[0].name: dummy_input}
        ort_outputs = session.run(None, ort_inputs)
        report["onnx_inference"] = True
        report["output_shape"] = list(ort_outputs[0].shape)
    except Exception as e:
        report["valid"] = False
        report["errors"].append(f"ONNX Runtime inference failed: {str(e)}")
        return report
    
    # Compare with PyTorch model
    if pytorch_model is not None:
        pytorch_model.eval()
        with torch.no_grad():
            pt_input = torch.tensor(dummy_input)
            pt_output = pytorch_model(pt_input).numpy()
        
        import numpy as np
        max_diff = np.max(np.abs(pt_output - ort_outputs[0]))
        report["max_difference"] = float(max_diff)
        report["outputs_match"] = max_diff < tolerance
        
        if max_diff >= tolerance:
            report["errors"].append(
                f"Output mismatch: max diff = {max_diff:.6f} (tolerance: {tolerance})"
            )
    
    # File size
    report["file_size_mb"] = os.path.getsize(onnx_path) / (1024 * 1024)
    
    return report
