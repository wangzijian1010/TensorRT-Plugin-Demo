import torch
import torch.nn as nn
import numpy as np
import onnx
import os

class LayerNormModel(nn.Module):
    """
    A simple model that contains only a LayerNorm operation.
    This can be used to test the TensorRT LayerNorm plugin.
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        """
        Args:
            normalized_shape: input shape from an expected input of size
            eps: a value added to the denominator for numerical stability
            elementwise_affine: a boolean value that when set to True, 
                this module has learnable per-element affine parameters (gamma, beta)
        """
        super(LayerNormModel, self).__init__()
        self.layer_norm = nn.LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine
        )
    
    def forward(self, x):
        """
        Forward pass of the LayerNorm model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after LayerNorm operation
        """
        return self.layer_norm(x)


def export_layernorm_onnx(hidden_size=768, batch_size=1, seq_length=384, output_file="layernorm_model.onnx"):
    """
    Export a PyTorch LayerNorm model to ONNX format
    
    Args:
        hidden_size: Size of the hidden dimension
        batch_size: Batch size for the model
        seq_length: Sequence length for the model
        output_file: Path to save the ONNX model
    
    Returns:
        Path to the exported ONNX model
    """
    # Create model instance
    model = LayerNormModel(normalized_shape=hidden_size).eval()
    
    # Initialize model weights to make output deterministic
    with torch.no_grad():
        if hasattr(model.layer_norm, 'weight') and model.layer_norm.weight is not None:
            nn.init.ones_(model.layer_norm.weight)
        if hasattr(model.layer_norm, 'bias') and model.layer_norm.bias is not None:
            nn.init.zeros_(model.layer_norm.bias)
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, seq_length, hidden_size)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence'},
            'output': {0: 'batch_size', 1: 'sequence'}
        },
        opset_version=11,
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"Model exported to {output_file}")
    return output_file


def test_layernorm_model(hidden_size=768, batch_size=2, seq_length=4):
    """
    Test the PyTorch LayerNorm model with a sample input
    """
    # Create model instance
    model = LayerNormModel(normalized_shape=hidden_size).eval()
    
    # Generate sample input
    sample_input = torch.randn(batch_size, seq_length, hidden_size)
    
    # Run forward pass
    with torch.no_grad():
        output = model(sample_input)
    
    # Print shape information
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Compute manually for verification
    input_data = sample_input.detach().numpy()
    # Calculate mean along last dimension
    mean = np.mean(input_data, axis=-1, keepdims=True)
    # Calculate variance along last dimension
    var = np.var(input_data, axis=-1, keepdims=True)
    # Apply normalization
    normalized = (input_data - mean) / np.sqrt(var + 1e-5)
    
    # Compare with PyTorch implementation
    torch_output = output.detach().numpy()
    diff = np.abs(normalized - torch_output).max()
    print(f"Maximum difference between manual computation and PyTorch: {diff:.6f}")


if __name__ == "__main__":
    # Test the LayerNorm model with small dimensions for verification
    print("Testing LayerNorm model:")
    test_layernorm_model(hidden_size=8, batch_size=2, seq_length=4)
    
    # Export the model to ONNX format with more realistic dimensions
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "layernorm_model.onnx")
    export_layernorm_onnx(
        hidden_size=768,  # Common hidden size in transformer models
        batch_size=1, 
        seq_length=384,   # Common sequence length
        output_file=output_path
    )