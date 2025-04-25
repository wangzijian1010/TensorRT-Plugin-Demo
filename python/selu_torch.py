import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
import numpy as np
import time
import os
import onnx


# Use PyTorch's built-in SELU implementation
class StandardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        self.selu = nn.SELU()  # Use PyTorch's built-in SELU
        self.conv.weight.data.fill_(1)
        self.conv.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv(x)
        x = self.selu(x)
        return x


def export_and_analyze_onnx():
    """导出ONNX模型并分析其结构"""
    print("开始导出和分析ONNX模型...")
    
    # 确保输出目录存在
    os.makedirs("/home/TensorRT_Plugin_Demo/workspace", exist_ok=True)
    onnx_path = "/home/TensorRT_Plugin_Demo/workspace/selu_torch.onnx"
    
    # 创建模型和示例输入
    model = StandardModel().eval()
    dummy_input = torch.zeros(1, 1, 3, 3)
    
    # 导出ONNX模型
    print(f"导出ONNX模型到：{onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        verbose=True,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"}
        },
    )
    
    # 加载并分析ONNX模型结构
    print("\n加载ONNX模型并分析结构...")
    onnx_model = onnx.load(onnx_path)
    
    print("\n===== ONNX模型结构 =====")
    print(f"图名称：{onnx_model.graph.name}")
    print(f"节点数量：{len(onnx_model.graph.node)}")
    
    # 输出每个节点的详细信息
    print("\n节点详情：")
    for i, node in enumerate(onnx_model.graph.node):
        print(f"\n节点 {i+1}: {node.op_type}")
        print(f"  输入: {', '.join(node.input)}")
        print(f"  输出: {', '.join(node.output)}")
        
        if node.attribute:
            print("  属性:")
            for attr in node.attribute:
                if hasattr(attr, 'f'):
                    print(f"    {attr.name}: {attr.f}")
                elif hasattr(attr, 'i'):
                    print(f"    {attr.name}: {attr.i}")
                elif hasattr(attr, 's'):
                    print(f"    {attr.name}: {attr.s}")
                else:
                    print(f"    {attr.name}: (复杂类型)")
    
    print("\n===== 输入输出信息 =====")
    print("输入:")
    for inp in onnx_model.graph.input:
        print(f"  名称: {inp.name}")
        try:
            shape = [dim.dim_value if hasattr(dim, 'dim_value') and dim.dim_value else '?' 
                     for dim in inp.type.tensor_type.shape.dim]
            print(f"  形状: {shape}")
        except:
            print("  形状: 未知")
    
    print("\n输出:")
    for out in onnx_model.graph.output:
        print(f"  名称: {out.name}")
        try:
            shape = [dim.dim_value if hasattr(dim, 'dim_value') and dim.dim_value else '?' 
                     for dim in out.type.tensor_type.shape.dim]
            print(f"  形状: {shape}")
        except:
            print("  形状: 未知")
    
    print("\nONNX模型分析完成！")
    return onnx_path

def check_onnx_execution(onnx_path):
    """使用ONNX Runtime检验模型执行结果"""
    try:
        import onnxruntime as ort
        print("\n===== 使用ONNX Runtime测试模型 =====")
        
        # 创建测试输入
        test_input = torch.ones(1, 1, 3, 3)
        
        # PyTorch模型执行
        torch_model = StandardModel().eval()
        with torch.no_grad():
            torch_output = torch_model(test_input).numpy()
        
        # ONNX Runtime执行
        print("初始化ONNX Runtime会话...")
        session = ort.InferenceSession(onnx_path)
        ort_inputs = {session.get_inputs()[0].name: test_input.numpy()}
        ort_output = session.run(None, ort_inputs)[0]
        
        # 比较结果
        diff = np.abs(torch_output - ort_output).max()
        print(f"PyTorch和ONNX Runtime输出最大差异: {diff}")
        print(f"测试{'通过' if np.allclose(torch_output, ort_output) else '失败'}")
        
        # 显示原始值与SELU结果
        print("\n输入值:")
        print(test_input.numpy())
        print("\nPyTorch SELU输出:")
        print(torch_output)
        print("\nONNX Runtime输出:")
        print(ort_output)
        
    except ImportError:
        print("警告: 无法导入onnxruntime，跳过执行测试")

if __name__ == "__main__":
    onnx_path = export_and_analyze_onnx()
    check_onnx_execution(onnx_path)