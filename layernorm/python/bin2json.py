import numpy as np
import json
import os

def bin_to_json(bin_file, output_json, shape=(1, 10, 64), dtype=np.float32):
    """
    将二进制.bin文件转换为NumPy数组并保存为JSON
    
    参数:
        bin_file: 输入的二进制文件路径
        output_json: 输出JSON文件路径
        shape: 数据的形状，默认为(1, 10, 64)
        dtype: 数据类型，默认为np.float32
    """
    # 检查文件是否存在
    if not os.path.exists(bin_file):
        print(f"错误: 找不到文件 {bin_file}")
        return False
    
    # 从二进制文件加载数据
    data = np.fromfile(bin_file, dtype=dtype)
    
    # 检查数据大小是否匹配预期形状
    expected_size = np.prod(shape)
    if data.size != expected_size:
        print(f"警告: 数据大小({data.size})与预期形状{shape}不匹配(预期大小: {expected_size})")
        print(f"尝试自动推断形状...")
        
        # 假设只有最后一个维度未知，尝试推断
        if data.size % (shape[0] * shape[1]) == 0:
            last_dim = data.size // (shape[0] * shape[1])
            shape = (shape[0], shape[1], last_dim)
            print(f"推断形状为: {shape}")
        else:
            print("无法自动推断形状，使用一维数组")
            shape = (data.size,)
    
    # 重塑数组
    try:
        data = data.reshape(shape)
    except ValueError as e:
        print(f"重塑数组失败: {e}")
        print("使用一维数组")
        shape = (data.size,)
        data = data.reshape(shape)
    
    # 转换为可序列化的形式
    output_data = {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "data": data.tolist()  # 将NumPy数组转换为Python列表
    }
    
    # 保存为JSON
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"成功将 {bin_file} 转换为 {output_json}")
    print(f"形状: {data.shape}, 类型: {data.dtype}")
    
    return True

# 使用示例:
if __name__ == "__main__":
    # 将TensorRT输出的二进制文件转换为JSON
    input_bin = "/home/TensorRT_Plugin_Demo/layernorm/model_input_output_output.bin"
    # output_bin = "output_0.bin"  # trtexec生成的输出文件名通常是output_0.bin
    
    # 转换输入文件
    bin_to_json(input_bin, "/home/TensorRT_Plugin_Demo/layernorm/output_py.json", shape=(1, 10, 64))
    
