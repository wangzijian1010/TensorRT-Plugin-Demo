import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import os
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import onnxruntime as ort

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

# For reference - the built-in SELU uses these parameters:
# Default alpha=1.6732632423543772
# Default scale=1.0507009873554805
# And the formula: scale * (max(0,x) + min(0, alpha * (exp(x) - 1)))

class MYSELUImpl(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, p):
        return g.op("MYSELU", x, p, 
                   alpha_f=1.0, 
                   beta_f=1.0,  
                   domain_s="")

    @staticmethod
    def forward(ctx, x, p):
        return x * 1 / (1 + torch.exp(-x))

class MYSELU(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.param = nn.Parameter(torch.arange(n).float())

    def forward(self, x):
        return MYSELUImpl.apply(x, self.param)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        self.myselu = MYSELU(3)
        self.conv.weight.data.fill_(1)
        self.conv.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv(x)
        x = self.myselu(x)
        return x

def load_plugin_library():
    """加载TensorRT插件库"""
    plugin_path = "/home/TensorRT_Plugin_Demo/libmyselu_plugin.so" 
    if os.path.exists(plugin_path):
        import ctypes
        ctypes.CDLL(plugin_path, mode=ctypes.RTLD_GLOBAL)
        print(f"成功加载插件库: {plugin_path}")
    else:
        print(f"警告: 插件库不存在: {plugin_path}")

# TensorRT 10.1兼容的推理类
class TensorRTInfer:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # 加载引擎文件
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
        # 初始化形状变量
        self.input_shape = None
        self.output_shape = None
        
        # 计算绑定数量（TensorRT 10.1兼容）
        self.num_bindings = 0
        self.binding_names = []
        
        # 调试信息
        print("开始枚举TensorRT绑定...")
        
        while True:
            try:
                name = self.engine.get_binding_name(self.num_bindings)
                self.binding_names.append(name)
                print(f"发现绑定 {self.num_bindings}: {name}")
                self.num_bindings += 1
            except Exception as e:
                print(f"绑定枚举结束，共找到 {self.num_bindings} 个绑定")
                break
        
        if self.num_bindings == 0:
            raise RuntimeError("没有找到任何绑定，TensorRT引擎可能有问题")
        
        # 获取输入/输出索引
        self.input_idx = -1
        self.output_idx = -1
        self.bindings = []
        
        for i in range(self.num_bindings):
            try:
                name = self.engine.get_binding_name(i)
                is_input = self.engine.binding_is_input(i)
                dims = self.engine.get_binding_dimensions(i)
                dtype = self.engine.get_binding_dtype(i)
                
                print(f"绑定 {i}: 名称={name}, 是否输入={is_input}, 形状={dims}")
                
                if is_input:
                    self.input_idx = i
                    self.input_shape = tuple(dims)
                    print(f"将绑定 {i} 设为输入, 形状={self.input_shape}")
                else:
                    self.output_idx = i
                    self.output_shape = tuple(dims)
                    print(f"将绑定 {i} 设为输出, 形状={self.output_shape}")
                
                # 分配GPU内存
                shape = tuple(dims)
                dtype_np = trt.nptype(dtype)
                size = trt.volume(shape) * dtype_np().itemsize
                self.bindings.append(cuda.mem_alloc(size))
                print(f"为绑定 {i} 分配了 {size} 字节的GPU内存")
                
            except Exception as e:
                print(f"处理绑定 {i} 时出错: {e}")
                raise
        
        # 验证输入和输出是否正确设置
        if self.input_idx == -1:
            raise RuntimeError("找不到输入绑定")
        if self.output_idx == -1:
            raise RuntimeError("找不到输出绑定")
        if self.input_shape is None:
            raise RuntimeError("输入形状未设置")
        if self.output_shape is None:
            raise RuntimeError("输出形状未设置")
            
    def infer(self, input_data):
        print(f"推理: 输入形状={input_data.shape}, 输出形状={self.output_shape}")
        
        # 验证输入形状
        if len(input_data.shape) != len(self.input_shape):
            raise ValueError(f"输入维度数不匹配: 期望 {len(self.input_shape)}, 实际 {len(input_data.shape)}")
        
        # 如果输出形状有0维度（未知维度），则使用输入的对应维度
        output_shape = list(self.output_shape)
        for i in range(len(output_shape)):
            if output_shape[i] == 0 and i < len(input_data.shape):
                output_shape[i] = input_data.shape[i]
        
        # 准备输入数据
        host_input = np.ascontiguousarray(input_data)
        host_output = np.empty(output_shape, dtype=np.float32)
        
        print(f"准备执行: 输入={host_input.shape}, 输出={host_output.shape}")
        
        # 拷贝输入数据到GPU
        cuda.memcpy_htod(self.bindings[self.input_idx], host_input)
        
        # 执行推理
        success = self.context.execute(self.num_bindings, self.bindings)
        if not success:
            raise RuntimeError("TensorRT推理执行失败")
        
        # 拷贝输出数据到CPU
        cuda.memcpy_dtoh(host_output, self.bindings[self.output_idx])
        
        return host_output
        
    def __del__(self):
        # 清理资源
        for binding in self.bindings:
            binding.free()

def build_engine(onnx_path, engine_path):
    """构建TensorRT引擎"""
    logger = trt.Logger(trt.Logger.VERBOSE)  # 使用详细日志级别
    builder = trt.Builder(logger)
    
    # TensorRT 10.1兼容的方式创建网络
    print("创建TensorRT网络...")
    network = builder.create_network(1 << 0)  # EXPLICIT_BATCH = 1 << 0
    
    parser = trt.OnnxParser(network, logger)
    
    # 解析ONNX文件
    print(f"解析ONNX文件: {onnx_path}")
    with open(onnx_path, 'rb') as model:
        model_bytes = model.read()
        if not parser.parse(model_bytes):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    # 打印网络信息
    print(f"网络层数: {network.num_layers}")
    print(f"网络输入数: {network.num_inputs}")
    print(f"网络输出数: {network.num_outputs}")
    
    # 配置构建器
    print("配置TensorRT构建器...")
    builder.max_workspace_size = 1 << 30  # 1GB
    builder.max_batch_size = 2
    
    # 构建引擎
    print("构建TensorRT引擎...")
    engine = builder.build_cuda_engine(network)
    if engine is None:
        print("ERROR: 引擎构建失败")
        return False
    
    # 打印引擎信息
    print(f"引擎绑定数: {engine.num_bindings if hasattr(engine, 'num_bindings') else '未知'}")
    
    # 保存引擎
    print(f"保存TensorRT引擎到: {engine_path}")
    with open(engine_path, 'wb') as f:
        serialized_engine = engine.serialize()
        f.write(serialized_engine)
    
    print(f"TensorRT引擎已保存，大小: {len(serialized_engine) / (1024*1024):.2f} MB")
    return True

def compare_outputs(torch_output, onnx_output, trt_output, rtol=1e-5, atol=1e-5):
    """比较不同模型的输出"""
    # 转换为numpy数组
    torch_output = torch_output.cpu().numpy()
    
    # 计算差异
    torch_onnx_diff = np.abs(torch_output - onnx_output).max() if onnx_output is not None else None
    torch_trt_diff = np.abs(torch_output - trt_output).max() if trt_output is not None else None
    onnx_trt_diff = np.abs(onnx_output - trt_output).max() if onnx_output is not None and trt_output is not None else None
    
    print("\n===== 输出比较 =====")
    if torch_onnx_diff is not None:
        print(f"PyTorch 和 ONNX 最大差异: {torch_onnx_diff}")
    if torch_trt_diff is not None:
        print(f"PyTorch 和 TensorRT 最大差异: {torch_trt_diff}")
    if onnx_trt_diff is not None:
        print(f"ONNX 和 TensorRT 最大差异: {onnx_trt_diff}")
    
    # 验证是否在误差范围内
    torch_onnx_match = False
    torch_trt_match = False
    onnx_trt_match = False
    
    if onnx_output is not None:
        torch_onnx_match = np.allclose(torch_output, onnx_output, rtol=rtol, atol=atol)
    if trt_output is not None:
        torch_trt_match = np.allclose(torch_output, trt_output, rtol=rtol, atol=atol)
    if onnx_output is not None and trt_output is not None:
        onnx_trt_match = np.allclose(onnx_output, trt_output, rtol=rtol, atol=atol)
    
    print("\n===== 验证结果 =====")
    if onnx_output is not None:
        print(f"PyTorch 和 ONNX 一致: {'✓' if torch_onnx_match else '✗'}")
    if trt_output is not None:
        print(f"PyTorch 和 TensorRT 一致: {'✓' if torch_trt_match else '✗'}")
    if onnx_output is not None and trt_output is not None:
        print(f"ONNX 和 TensorRT 一致: {'✓' if onnx_trt_match else '✗'}")
    
    return torch_trt_match  # 主要关注PyTorch和TensorRT结果是否一致

def main():
    # 确保workspace目录存在
    os.makedirs("workspace", exist_ok=True)
    
    # 定义文件路径
    onnx_model_path = "/home/TensorRT_Plugin_Demo/demo.onnx"
    trt_engine_path = "/home/TensorRT_Plugin_Demo/demo.engine"
    std_onnx_model_path = "/home/TensorRT_Plugin_Demo/std_demo.onnx"
    
    # 创建模型并设置为eval模式
    model = Model().eval()
    std_model = StandardModel().eval()

    # 创建输入数据
    input_data = torch.tensor([
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]],
        [[-1, 1, 1],
         [1, 0, 1],
         [1, 1, -1]]
    ], dtype=torch.float32).view(2, 1, 3, 3)

    input_numpy = input_data.numpy()
    input_numpy.tofile("/home/TensorRT_Plugin_Demo/data.bin")

    # 1. PyTorch模型推理 (自定义MYSELU和标准SELU)
    with torch.no_grad():
        torch_output = model(input_data)
        std_torch_output = std_model(input_data)
    
    print("PyTorch自定义MYSELU输出:\n", torch_output)
    print("PyTorch标准SELU输出:\n", std_torch_output)
    
    # 2. 导出标准SELU的ONNX模型
    if not os.path.exists(std_onnx_model_path):
        # 创建dummy输入
        dummy = torch.zeros(1, 1, 3, 3)
        
        # 导出ONNX
        torch.onnx.export(
            std_model,
            (dummy,),
            std_onnx_model_path,
            verbose=False,
            input_names=["image"],
            output_names=["output"],
            opset_version=11,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            keep_initializers_as_inputs=True,
            dynamic_axes={
                "image": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "height", 3: "width"}
            }
        )
        print(f"标准SELU的ONNX模型已导出到: {std_onnx_model_path}")
    
    # 3. 使用ONNX Runtime执行标准SELU模型
    try:
        std_ort_session = ort.InferenceSession(std_onnx_model_path)
        std_ort_inputs = {std_ort_session.get_inputs()[0].name: input_data.numpy()}
        std_onnx_output = std_ort_session.run(None, std_ort_inputs)[0]
        print("标准SELU的ONNX Runtime输出:\n", std_onnx_output)
        
        # 检查标准SELU的输出是否与PyTorch一致
        std_match = np.allclose(std_torch_output.cpu().numpy(), std_onnx_output, rtol=1e-5, atol=1e-5)
        print(f"标准SELU的PyTorch和ONNX输出一致: {'✓' if std_match else '✗'}")
        
        # 打印ONNX模型的操作符结构
        import onnx
        std_model_onnx = onnx.load(std_onnx_model_path)
        print("\n===== 标准SELU的ONNX模型结构 =====")
        for node in std_model_onnx.graph.node:
            print(f"操作符: {node.op_type}, 输入: {node.input}, 输出: {node.output}")
            if node.attribute:
                print("  属性:")
                for attr in node.attribute:
                    print(f"    {attr.name}: {attr}")
        
    except Exception as e:
        print(f"标准SELU的ONNX Runtime执行失败: {e}")
    
    # 导出自定义MYSELU的ONNX模型
    if not os.path.exists(onnx_model_path):
        # 创建dummy输入
        dummy = torch.zeros(1, 1, 3, 3)
        
        # 注册自定义算子的符号函数（确保在导出前注册）
        from torch.onnx import register_custom_op_symbolic
        register_custom_op_symbolic('::MYSELU', MYSELUImpl.symbolic, 11)
        
        # 导出ONNX
        torch.onnx.export(
            model,
            (dummy,),
            onnx_model_path,
            verbose=False,
            input_names=["image"],
            output_names=["output"],
            opset_version=11,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            keep_initializers_as_inputs=True,
            dynamic_axes={
                "image": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "height", 3: "width"}
            },
            enable_onnx_checker=False
        )
        print(f"自定义MYSELU的ONNX模型已导出到: {onnx_model_path}")
        
        # 打印ONNX模型的操作符结构
        import onnx
        model_onnx = onnx.load(onnx_model_path)
        print("\n===== 自定义MYSELU的ONNX模型结构 =====")
        for node in model_onnx.graph.node:
            print(f"操作符: {node.op_type}, 输入: {node.input}, 输出: {node.output}")
            if node.attribute:
                print("  属性:")
                for attr in node.attribute:
                    print(f"    {attr.name}: {attr}")

    # 3. 使用ONNX Runtime执行ONNX模型
    try:
        ort_session = ort.InferenceSession(onnx_model_path)
        ort_inputs = {ort_session.get_inputs()[0].name: input_data.numpy()}
        onnx_output = ort_session.run(None, ort_inputs)[0]
        print("ONNX Runtime输出:\n", onnx_output)
    except Exception as e:
        print(f"ONNX Runtime执行失败: {e}")
        onnx_output = None
    
    # 4. 加载TensorRT插件库（在构建引擎前）
    load_plugin_library()
    
    # 5. 构建TensorRT引擎（如果不存在）
    if not os.path.exists(trt_engine_path):
        print(f"TensorRT引擎不存在，开始构建...")
        success = build_engine(onnx_model_path, trt_engine_path)
        if not success:
            print("构建TensorRT引擎失败!")
            return
    else:
        print(f"使用已存在的TensorRT引擎: {trt_engine_path}")
    
    # 6. 使用TensorRT执行模型
    try:
        print("初始化TensorRT推理器...")
        trt_infer = TensorRTInfer(trt_engine_path)
        
        print("执行TensorRT推理...")
        trt_output = trt_infer.infer(input_data.numpy())
        print("TensorRT输出:\n", trt_output)
        
        # 7. 比较PyTorch和TensorRT的输出结果
        match = compare_outputs(torch_output, onnx_output, trt_output)
        if match:
            print("\n恭喜! PyTorch和TensorRT输出结果一致，自定义算子实现正确。")
        else:
            print("\n警告: 输出结果不一致，请检查自定义算子实现。")
            
    except Exception as e:
        import traceback
        print(f"TensorRT执行失败: {e}")
        traceback.print_exc()
        trt_output = None

if __name__ == "__main__":
    main()