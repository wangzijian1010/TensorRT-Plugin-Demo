import onnx_graphsurgeon as gs
import onnx
import numpy as np
import os

# 加载模型
model_path = "layernorm_model.onnx"
graph = gs.import_onnx(onnx.load(model_path))

# 找到我们关注的四个节点类型
pow_nodes = [node for node in graph.nodes if node.op == "Pow"]
reduce_mean_nodes = [node for node in graph.nodes if node.op == "ReduceMean"]
add_nodes = [node for node in graph.nodes if node.op == "Add"]
sqrt_nodes = [node for node in graph.nodes if node.op == "Sqrt"]

print(f"找到 {len(pow_nodes)} 个 Pow 节点")
print(f"找到 {len(reduce_mean_nodes)} 个 ReduceMean 节点")
print(f"找到 {len(add_nodes)} 个 Add 节点")
print(f"找到 {len(sqrt_nodes)} 个 Sqrt 节点")

# 找出符合特定模式的节点组合: Pow -> ReduceMean -> Add -> Sqrt
matched_groups = []

# 从Pow节点开始查找连接关系
for pow_node in pow_nodes:
    pow_output = pow_node.outputs[0]
    
    # 查找与Pow相连的ReduceMean节点
    connected_rm = None
    for rm_node in reduce_mean_nodes:
        if pow_output in rm_node.inputs:
            connected_rm = rm_node
            break
    
    if not connected_rm:
        continue  # 如果找不到连接的ReduceMean节点，跳过
    
    rm_output = connected_rm.outputs[0]
    
    # 查找与ReduceMean相连的Add节点
    connected_add = None
    for add_node in add_nodes:
        if rm_output in add_node.inputs:
            connected_add = add_node
            break
    
    if not connected_add:
        continue  # 如果找不到连接的Add节点，跳过
    
    add_output = connected_add.outputs[0]
    
    # 查找与Add相连的Sqrt节点
    connected_sqrt = None
    for sqrt_node in sqrt_nodes:
        if add_output in sqrt_node.inputs:
            connected_sqrt = sqrt_node
            break
    
    if not connected_sqrt:
        continue  # 如果找不到连接的Sqrt节点，跳过
    
    # 找到一组完整的模式
    matched_groups.append((pow_node, connected_rm, connected_add, connected_sqrt))
    print(f"找到匹配的模式:")
    print(f"  Pow: {pow_node.name}")
    print(f"  ReduceMean: {connected_rm.name}")
    print(f"  Add: {connected_add.name}")
    print(f"  Sqrt: {connected_sqrt.name}")

# 替换找到的模式
for i, (pow_node, rm_node, add_node, sqrt_node) in enumerate(matched_groups):
    # 确定新节点的输入
    inputs = []
    
    # 收集Pow节点的输入作为自定义节点的输入
    for inp in pow_node.inputs:
        if inp not in inputs:
            inputs.append(inp)
    
    # 如果有任何额外的常量输入，也要包含它们
    for node in [rm_node, add_node]:
        for inp in node.inputs:
            if inp.is_constant and inp not in inputs:
                inputs.append(inp)
    
    # 确定新节点的输出
    outputs = sqrt_node.outputs
    
    # 创建自定义节点
    custom_node = gs.Node(
        op="CustomTest",          # 操作类型
        name=f"CustomTest_{i}",   # 节点名称
        inputs=inputs,            # 输入
        outputs=outputs,          # 输出
        attrs={                   # 属性
            "domain_s": "com.example",
            "epsilon_f": 1e-5,    # 可能的参数
        }
    )
    
    print(f"创建自定义节点 CustomTest_{i}")
    print(f"  输入: {[inp.name for inp in inputs if inp is not None]}")
    print(f"  输出: {[out.name for out in outputs if out is not None]}")
    
    # 移除旧节点
    graph.nodes.remove(pow_node)
    graph.nodes.remove(rm_node)
    graph.nodes.remove(add_node)
    graph.nodes.remove(sqrt_node)
    
    # 添加新节点
    graph.nodes.append(custom_node)
    
    print(f"已将模式替换为 CustomTest_{i}")

# 清理图并保存
graph.cleanup()
output_file = "./python/layernorm_custom_model.onnx"
onnx.save(gs.export_onnx(graph), output_file)
print(f"已保存修改后的模型到 {output_file}")