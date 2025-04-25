import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import ctypes
from numpy.testing import assert_allclose
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd

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

class EngineRunner:
    def __init__(self, engine_path, plugin_path):
        try:
            # 加载插件
            ctypes.CDLL(plugin_path)
        except OSError as e:
            raise RuntimeError(f"❌ Failed to load plugin: {plugin_path}. Error: {str(e)}")
        
        # 创建 TensorRT 运行环境
        logger = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(logger)
        
        # 加载 engine
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # 初始化内存相关变量
        self.stream = cuda.Stream()
        self.bindings = []
        self.host_inputs = []
        self.host_outputs = []
        self.device_inputs = []
        self.device_outputs = []
        self.input_shapes = []
        self.output_shapes = []
        
        # 打印引擎信息
        print("TensorRT Engine Info:")
        num_io_tensors = 0
        try:
            # 在TensorRT 10.1中可能没有num_io_tensors属性
            # 尝试获取绑定数量
            while True:
                name = self.engine.get_tensor_name(num_io_tensors)
                num_io_tensors += 1
        except:
            pass
        
        print(f"- Number of IO tensors: {num_io_tensors}")
        
        # 查找并处理所有 tensor
        input_count = 0
        output_count = 0
        
        for i in range(num_io_tensors):
            try:
                name = self.engine.get_tensor_name(i)
                dims = self.engine.get_binding_dimensions(i)
                is_input = self.engine.binding_is_input(i)
                
                print(f"- Tensor {i}: {name}, shape: {dims}, {'input' if is_input else 'output'}")
                
                # 记录形状信息
                shape = [dim for dim in dims]
                if is_input:
                    self.input_shapes.append(shape)
                    input_count += 1
                else:
                    self.output_shapes.append(shape)
                    output_count += 1
            except Exception as e:
                print(f"Warning: Failed to process tensor at index {i}: {str(e)}")
        
        print(f"TensorRT Engine has {input_count} inputs and {output_count} outputs")
        
        # 推迟内存分配到实际推理时
        self.memory_allocated = False
    
    def allocate_memory(self, input_data):
        """延迟分配内存，使用实际的输入数据"""
        if self.memory_allocated:
            return
        
        print("Allocating memory for inputs...")
        
        try:
            # 清除之前可能已经分配的内存
            self.host_inputs = []
            self.host_outputs = []
            self.device_inputs = []
            self.device_outputs = []
            self.bindings = []
            
            # 处理所有输入
            input_binding_idxs = []
            for i in range(len(input_data)):
                # 找到对应的输入绑定索引
                binding_idx = -1
                for idx in range(self.engine.num_bindings):
                    if self.engine.binding_is_input(idx):
                        if len(input_binding_idxs) == i:  # 找到第i个输入绑定
                            binding_idx = idx
                            input_binding_idxs.append(idx)
                            break
                
                if binding_idx == -1:
                    raise RuntimeError(f"Could not find binding for input {i}")
                
                # 设置输入形状
                input_shape = input_data[i].shape
                print(f"Input {i} shape: {input_shape}")
                
                # 计算内存大小
                size = int(np.prod(input_shape))
                dtype = np.dtype(np.float32)  # 假设输入是float32
                
                # 分配内存
                try:
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(size * dtype.itemsize)
                except Exception as e:
                    print(f"Failed to allocate memory for input {i}: {str(e)}")
                    print(f"Size: {size}, Shape: {input_shape}, Type: {dtype}")
                    raise
                
                self.host_inputs.append(host_mem)
                self.device_inputs.append(device_mem)
                # 确保绑定列表足够长
                while len(self.bindings) <= binding_idx:
                    self.bindings.append(None)
                self.bindings[binding_idx] = int(device_mem)
            
            # 处理所有输出
            print("Allocating memory for outputs...")
            output_binding_idxs = []
            for idx in range(self.engine.num_bindings):
                if not self.engine.binding_is_input(idx) and idx not in output_binding_idxs:
                    output_binding_idxs.append(idx)
            
            for i, binding_idx in enumerate(output_binding_idxs):
                # 获取输出形状
                dims = self.engine.get_binding_dimensions(binding_idx)
                # 对于动态形状，使用输入的批次大小
                output_shape = []
                for j, dim in enumerate(dims):
                    if dim == -1 and j == 0:  # 批次维度为动态
                        output_shape.append(input_data[0].shape[0])
                    else:
                        output_shape.append(dim)
                
                print(f"Output {i} shape: {output_shape}")
                
                # 计算内存大小
                size = int(np.prod(output_shape))
                dtype = np.dtype(np.float32)  # 假设输出是float32
                
                # 分配内存
                try:
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(size * dtype.itemsize)
                except Exception as e:
                    print(f"Failed to allocate memory for output {i}: {str(e)}")
                    print(f"Size: {size}, Shape: {output_shape}, Type: {dtype}")
                    raise
                
                self.host_outputs.append(host_mem)
                self.device_outputs.append(device_mem)
                # 确保绑定列表足够长
                while len(self.bindings) <= binding_idx:
                    self.bindings.append(None)
                self.bindings[binding_idx] = int(device_mem)
                
                # 保存输出形状以便reshape
                self.output_shapes.append(output_shape)
            
            # 检查是否所有绑定都已设置
            for i, binding in enumerate(self.bindings):
                if binding is None:
                    raise RuntimeError(f"Binding at index {i} was not set")
            
            self.memory_allocated = True
            print("Memory allocation successful")
        
        except Exception as e:
            print(f"Error during memory allocation: {str(e)}")
            self.free_memory()
            raise
    
    def free_memory(self):
        """释放所有分配的内存"""
        try:
            for mem in self.device_inputs + self.device_outputs:
                if mem:
                    mem.free()
            
            self.host_inputs = []
            self.host_outputs = []
            self.device_inputs = []
            self.device_outputs = []
            self.bindings = []
            self.memory_allocated = False
            print("Memory freed successfully")
        except Exception as e:
            print(f"Error during memory freeing: {str(e)}")
    
    def infer(self, inputs):
        """执行推理"""
        try:
            # 延迟分配内存直到知道实际输入数据
            self.allocate_memory(inputs)
            
            # 复制输入数据到 GPU
            for i, input_data in enumerate(inputs):
                # 将输入数据扁平化
                flat_input = input_data.flatten()
                if len(flat_input) > len(self.host_inputs[i]):
                    raise ValueError(f"Input {i} size mismatch: buffer size {len(self.host_inputs[i])}, data size {len(flat_input)}")
                
                # 复制数据
                self.host_inputs[i][:len(flat_input)] = flat_input
                cuda.memcpy_htod_async(self.device_inputs[i], self.host_inputs[i], self.stream)
            
            # 执行推理 - TensorRT 10.1版本兼容的方式
            success = self.context.execute(self.engine.num_bindings, self.bindings)
            if not success:
                raise RuntimeError("Failed to execute inference")
            
            # 复制输出数据到 CPU 并重塑
            outputs = []
            for i, shape in enumerate(self.output_shapes):
                # 从设备复制到主机
                cuda.memcpy_dtoh_async(self.host_outputs[i], self.device_outputs[i], self.stream)
                
                # 同步流
                self.stream.synchronize()
                
                # 创建适当形状的输出数组
                output_size = int(np.prod(shape))
                output = self.host_outputs[i][:output_size].reshape(shape)
                outputs.append(output)
            
            return outputs
        
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def __del__(self):
        # 清理资源
        self.free_memory()
        try:
            del self.context
            del self.engine
            del self.stream
        except:
            pass

def compare_outputs(torch_output, trt_output, rtol=1e-5, atol=1e-5):
    """比较输出结果"""
    # 确保PyTorch输出是NumPy数组
    if isinstance(torch_output, torch.Tensor):
        torch_output = torch_output.detach().cpu().numpy()
    
    try:
        assert_allclose(torch_output, trt_output, rtol=rtol, atol=atol)
        print("✅ Outputs match within tolerance!")
        print(f"Max absolute difference: {np.max(np.abs(torch_output - trt_output))}")
        print(f"Mean absolute difference: {np.mean(np.abs(torch_output - trt_output))}")
        return True
    except AssertionError as e:
        print("❌ Outputs do not match!")
        print(f"Error: {str(e)}")
        print(f"Max absolute difference: {np.max(np.abs(torch_output - trt_output))}")
        print(f"Mean absolute difference: {np.mean(np.abs(torch_output - trt_output))}")
        return False

def main():
    # 设置路径
    engine_path = "/home/TensorRT_Plugin_Demo/demo.engine"
    plugin_path = "/home/TensorRT_Plugin_Demo/libmyselu_plugin.so"
    
    # 创建PyTorch模型
    model = Model().eval()
    
    # 生成测试数据 - 使用与模型匹配的形状
    input_shape = (2, 1, 3, 3)  # 根据Model类中的设计调整
    input_data = np.array([
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]],
        [[-1, 1, 1],
         [1, 0, 1],
         [1, 1, -1]]
    ], dtype=np.float32).reshape(2, 1, 3, 3)
    
    # 运行PyTorch模型
    print("Running PyTorch model...")
    torch_input = torch.tensor(input_data)
    with torch.no_grad():
        torch_output = model(torch_input)
    print("PyTorch output shape:", torch_output.shape)
    print("PyTorch output sample:", torch_output[0, 0, :3, :3])
    
    try:
        # 运行TensorRT engine
        print("\nRunning TensorRT engine...")
        runner = EngineRunner(engine_path, plugin_path)
        trt_outputs = runner.infer([input_data])
        
        if trt_outputs is None or len(trt_outputs) == 0:
            print("No output from TensorRT engine")
            return False
            
        trt_output = trt_outputs[0]
        print("TensorRT output shape:", trt_output.shape)
        print("TensorRT output sample:", trt_output[0, 0, :3, :3] if trt_output.ndim >= 3 else trt_output)
        
        # 比较结果
        print("\nComparing outputs...")
        success = compare_outputs(torch_output, trt_output)
        
        return success
    
    except Exception as e:
        print(f"Error during TensorRT execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Test passed successfully!")
        else:
            print("\n❌ Test failed!")
    except Exception as e:
        import traceback
        print(f"\n❌ Error occurred: {str(e)}")
        traceback.print_exc()