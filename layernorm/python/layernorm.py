import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
import os

# 定义含有Conv和LayerNorm的模型
class ConvLayerNormModel(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(ConvLayerNormModel, self).__init__()
        self.conv = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.layer_norm = nn.LayerNorm(normalized_shape=normalized_shape, 
                                       eps=eps, 
                                       elementwise_affine=elementwise_affine)
    
    def forward(self, x):
        # x: (batch_size, seq_len, channels)
        x = x.permute(0, 2, 1)   # 转成 (batch_size, channels, seq_len)
        x = self.conv(x)
        x = x.permute(0, 2, 1)   # 再转回来 (batch_size, seq_len, channels)
        x = self.layer_norm(x)
        return x

# 定义比较函数
def compare_func(model, dummy_input, output_file_prefix="model_input_output"):
    # 使用PyTorch运行模型
    torch_output = model(dummy_input).detach().numpy()

    # 保存输入和输出为原始二进制 .bin 文件
    input_file = f"{output_file_prefix}_input.bin"
    output_file = f"{output_file_prefix}_output.bin"
    
    # 保存输入为原始二进制格式
    dummy_input_numpy = dummy_input.numpy()
    dummy_input_numpy.tofile(input_file)
    
    # 保存输出为原始二进制格式
    torch_output.tofile(output_file)

    print(f"输入已保存到 {input_file} (大小: {os.path.getsize(input_file)} 字节)")
    print(f"输出已保存到 {output_file} (大小: {os.path.getsize(output_file)} 字节)")

    return input_file, output_file, torch_output


# 创建模型
model = ConvLayerNormModel(normalized_shape=64)
model.eval()

# 创建示例输入
dummy_input = torch.randn(1, 10, 64)

# 导出ONNX
torch.onnx.export(
    model,
    dummy_input,
    "conv_layernorm_model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size", 1: "sequence_length"}
    }
)

print("模型已成功导出为 conv_layernorm_model.onnx")

# 比较函数
input_file, output_file, torch_output = compare_func(model, dummy_input)

# 使用ONNX Runtime验证模型
session = ort.InferenceSession("conv_layernorm_model.onnx")
onnx_output = session.run(None, {"input": dummy_input.numpy()})[0]

# 验证输出是否匹配
np.testing.assert_allclose(torch_output, onnx_output, rtol=1e-5, atol=1e-5)
print("PyTorch和ONNX运行结果匹配，验证通过!")

# 打印模型详细信息
print(f"\n模型信息:")
print(f"- 输入形状: {dummy_input.shape}")
print(f"- 输出形状: {torch_output.shape}")
print(f"- Conv参数: in_channels=64, out_channels=64, kernel_size=1")
print(f"- LayerNorm参数: normalized_shape=64, eps=1e-5")
