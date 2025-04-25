import torch
import torch.nn as nn
import os

class JustSELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.selu = nn.SELU()
        
    def forward(self, x):
        return self.selu(x)

def main():
    # 创建一个只有SELU的简单模型
    model = JustSELU().eval()
    dummy_input = torch.randn(1, 3, 10, 10)
    
    # 确保输出目录存在
    os.makedirs("/home/TensorRT_Plugin_Demo/python", exist_ok=True)
    onnx_path = "/home/TensorRT_Plugin_Demo/python/just_selu.onnx"
    
    # 导出ONNX模型
    print(f"导出只含SELU的ONNX模型到：{onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        
        verbose=True,
        input_names=["input"],
        output_names=["output"],
        opset_version=11
    )
    
    print("导出完成！请使用Netron查看此模型，分析SELU的表示方式")
    
    # 打印模型的基本信息
    print("\n原始PyTorch模型:")
    print(model)
    
    # 打印SELU的具体参数
    print("\nSELU参数:")
    # SELU的标准参数
    print(f"alpha = {1.6732632423543772}")
    print(f"scale = {1.0507009873554805}")

if __name__ == "__main__":
    main()
