import onnx_graphsurgeon as gs
import onnx
import numpy as np

# 加载模型
graph = gs.import_onnx(onnx.load("layernorm_model.onnx"))

# 定义一个辅助函数来获取数据类型的字符串表示
def get_dtype_str(dtype):
    if dtype == np.float32:
        return "FP32 (float32)"
    elif dtype == np.float16:
        return "FP16 (float16)"
    elif dtype == np.int64:
        return "INT64 (int64)"
    elif dtype == np.int32:
        return "INT32 (int32)"
    else:
        return f"其他类型: {dtype}"

# 首先打印原始模型中节点的数据类型信息
print("===== 修改前的节点数据类型信息 =====")
for node in graph.nodes:
    if node.name == "/layer_norm/Pow":  # 只显示Pow节点的详细信息
        print(f"\n节点: {node.name} (操作类型: {node.op})")
        
        # 打印输入张量的数据类型
        print("  输入张量:")
        for i, inp in enumerate(node.inputs):
            if inp is not None:
                dtype_str = get_dtype_str(inp.dtype) if inp.dtype is not None else "未知类型"
                shape_str = str(inp.shape) if inp.shape is not None else "未知形状"
                print(f"    [{i}] 名称: {inp.name}, 数据类型: {dtype_str}, 形状: {shape_str}")
        
        # 打印输出张量的数据类型
        print("  输出张量:")
        for i, out in enumerate(node.outputs):
            if out is not None:
                dtype_str = get_dtype_str(out.dtype) if out.dtype is not None else "未知类型"
                shape_str = str(out.shape) if out.shape is not None else "未知形状"
                print(f"    [{i}] 名称: {out.name}, 数据类型: {dtype_str}, 形状: {shape_str}")

# 显示有明确类型的张量
print("\n===== 有明确数据类型的张量 =====")
typed_tensors = [(name, tensor) for name, tensor in graph.tensors().items() if tensor.dtype is not None]
for name, tensor in typed_tensors:
    print(f"张量: {name}, 类型: {get_dtype_str(tensor.dtype)}, 形状: {tensor.shape}")

# 打印统计信息
print("\n===== 修改前的模型统计信息 =====")
fp32_count = sum(1 for tensor in graph.tensors().values() if tensor.dtype == np.float32)
fp16_count = sum(1 for tensor in graph.tensors().values() if tensor.dtype == np.float16)
print(f"FP32 张量数量: {fp32_count}")
print(f"FP16 张量数量: {fp16_count}")

# ----- 开始修改数据类型 -----
print("\n===== 开始修改模型数据类型 =====")

# 修改所有明确类型的张量 (这是真正生效的方式)
modified_count = 0
for name, tensor in graph.tensors().items():
    if tensor.dtype == np.float32:
        print(f"将张量 {name} 从 FP32 转换为 FP16")
        tensor.dtype = np.float16
        modified_count += 1

print(f"总共修改了 {modified_count} 个张量的数据类型")

# 修改后检查有明确类型的张量
print("\n===== 修改后有明确数据类型的张量 =====")
typed_tensors = [(name, tensor) for name, tensor in graph.tensors().items() if tensor.dtype is not None]
for name, tensor in typed_tensors:
    print(f"张量: {name}, 类型: {get_dtype_str(tensor.dtype)}, 形状: {tensor.shape}")

# 打印修改后的统计信息
print("\n===== 修改后的模型统计信息 =====")
fp32_count = sum(1 for tensor in graph.tensors().values() if tensor.dtype == np.float32)
fp16_count = sum(1 for tensor in graph.tensors().values() if tensor.dtype == np.float16)
print(f"FP32 张量数量: {fp32_count}")
print(f"FP16 张量数量: {fp16_count}")

# 修改节点属性中可能的数据类型信息
for node in graph.nodes:
    # 某些节点可能在属性中也有数据类型信息
    if hasattr(node, "attrs") and node.attrs is not None:
        for attr_name, attr_value in node.attrs.items():
            if attr_name == "dtype" and attr_value == 1:  # 1通常代表float32
                print(f"修改节点 {node.name} 的dtype属性从 float32 到 float16")
                node.attrs[attr_name] = 10  # 10通常代表float16

# 保存修改后的模型
onnx.save(gs.export_onnx(graph), "layernorm_fp16_model.onnx")
print("\n已保存修改后的模型到 layernorm_fp16_model.onnx")

print("\n===== 数据类型修改原理说明 =====")
print("ONNX模型中数据类型情况:")
print("1. 明确类型: 通常只有模型的输入、输出和权重张量具有明确的数据类型")
print("2. 未知类型: 大多数中间计算结果在静态分析时是'未知类型'，会在运行时根据输入动态推导")
print("\n如何有效修改模型精度:")
print("1. 修改所有明确类型的张量 (输入、输出和权重)")
print("2. 当这些明确类型的张量被修改为FP16后，推理引擎会相应地使用FP16进行中间计算")
print("3. 这样就能实现整个模型的FP16推理，包括Pow等计算节点")
print("\n注意: 尽管图中很多中间节点显示为'未知类型'，实际运行时它们会根据输入类型来确定计算精度")