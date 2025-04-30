import onnx
import onnx_graphsurgeon as gs
import numpy as np
import onnxruntime as ort
import os

def add_all_nodes_outputs_to_model(model_path, output_model_path=None):
    """
    将ONNX模型中所有节点的输出添加为模型的全局输出
    
    参数:
    model_path: 输入模型路径
    output_model_path: 输出模型路径，如果为None则在原文件名加上_with_debug_outputs后缀
    
    返回:
    output_model_path: 修改后的模型保存路径
    """
    if output_model_path is None:
        base_name = os.path.splitext(model_path)[0]
        output_model_path = f"{base_name}_with_debug_outputs.onnx"
    
    # 加载ONNX模型
    graph = gs.import_onnx(onnx.load(model_path))
    
    # 创建一个新的输出列表，包含原始输出
    new_outputs = list(graph.outputs)
    
    # 遍历所有节点，为每个节点添加输出
    for node in graph.nodes:
        for output_tensor in node.outputs:
            # 跳过已经是图输出的张量
            if output_tensor in graph.outputs:
                continue
                
            # 确保输出张量有数据类型信息
            if output_tensor.dtype is None:
                # 首先尝试从输入继承数据类型
                if node.inputs and node.inputs[0].dtype is not None:
                    output_tensor.dtype = node.inputs[0].dtype
                else:
                    # 如果无法从输入继承，则默认设置为float32
                    output_tensor.dtype = np.float32
            
            # 将此输出添加为图的输出
            new_outputs.append(output_tensor)
            print(f"添加节点 {node.name} ({node.op}) 的输出: {output_tensor.name}")
    
    # 更新图的输出
    graph.outputs = new_outputs
    
    # 导出修改后的模型
    onnx.save(gs.export_onnx(graph), output_model_path)
    print(f"已将修改后的模型保存到 {output_model_path}")
    
    # 使用ONNX Shape Inference来获取每个节点输出的形状信息
    try:
        model = onnx.load(output_model_path)
        model = onnx.shape_inference.infer_shapes(model)
        
        # 打印所有输出的形状信息
        print("\n===== 所有节点的输出维度信息 =====")
        for output in model.graph.output:
            shape = []
            for dim in output.type.tensor_type.shape.dim:
                if dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append(dim.dim_value)
            print(f"输出 '{output.name}': 形状 = {shape}")
    except Exception as e:
        print(f"形状推断过程中发生错误: {e}")
        
    return output_model_path


def get_node_output_dimensions(model_path, target_node_name, input_shape=None):
    """
    获取指定节点的输出维度信息
    
    参数:
    model_path: 模型路径（应该是已添加了所有节点输出的模型）
    target_node_name: 目标节点名称
    input_shape: 输入形状，如[1,10,64]，如果不指定则尝试从模型中获取
    
    返回:
    output_info: 包含输出维度和数据统计的字典
    """
    # 加载模型
    session = ort.InferenceSession(model_path)
    
    # 获取模型输入信息
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    
    # 准备输入数据
    if input_shape is None:
        # 尝试从模型获取形状
        shape = input_info.shape
        # 处理动态维度
        input_shape = []
        for dim in shape:
            if isinstance(dim, int):
                input_shape.append(dim)
            else:
                input_shape.append(1)  # 对于动态维度，使用默认值1
        print(f"使用自动检测的输入形状: {input_shape}")
    
    # 创建随机输入数据
    input_data = np.random.randn(*input_shape).astype(np.float32)
    
    # 查找目标输出节点
    target_output = None
    for output in session.get_outputs():
        if target_node_name == output.name:
            target_output = output.name
            break
    
    # 如果没有精确匹配，尝试部分匹配
    if not target_output:
        for output in session.get_outputs():
            if target_node_name in output.name:
                target_output = output.name
                print(f"找到部分匹配的输出: {output.name}")
                break
    
    if not target_output:
        print(f"未找到名称为 '{target_node_name}' 或包含该名称的输出节点")
        print("可用的输出节点有:")
        for output in session.get_outputs():
            print(f" - {output.name}")
        return None
    
    # 运行模型获取特定输出
    outputs = session.run([target_output], {input_name: input_data})
    output = outputs[0]
    
    # 收集输出信息
    output_info = {
        'name': target_output,
        'shape': output.shape,
        'dtype': str(output.dtype),
        'min': float(np.min(output)),
        'max': float(np.max(output)),
        'mean': float(np.mean(output)),
        'std': float(np.std(output))
    }
    
    # 打印输出信息
    print(f"\n===== 节点 '{target_output}' 的输出维度信息 =====")
    print(f"形状: {output_info['shape']}")
    print(f"数据类型: {output_info['dtype']}")
    print(f"数值范围: [{output_info['min']}, {output_info['max']}]")
    print(f"均值: {output_info['mean']}")
    print(f"标准差: {output_info['std']}")
    
    return output_info


# 示例用法
if __name__ == "__main__":
    # 定义模型路径
    # original_model_path = "/home/TensorRT_Plugin_Demo/layernorm/python/conv_layernorm_model.onnx"
    
    # # 步骤1: 将所有节点输出添加到模型
    # debug_model_path = add_all_nodes_outputs_to_model(original_model_path)
    
    # 步骤2: 获取特定节点的输出维度信息
    # 如果您想查看Conv_output_0节点的输出维度
    node_info = get_node_output_dimensions("model_with_debug_outputs.onnx", "output", input_shape=[1, 10, 64])
    # 也可以查看其他节点
    # node_info = get_node_output_dimensions(debug_model_path, "Transpose_output_0", input_shape=[1, 10, 64])