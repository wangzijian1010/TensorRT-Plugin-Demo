import onnx
import onnx_graphsurgeon as gs

def find_layernorm_patterns(graph):
    """寻找符合LayerNorm模式的子图"""
    patterns = []

    for node in graph.nodes:
        if node.op != "ReduceMean":
            continue

        # 第一步 ReduceMean -> Sub
        sub_node = next(iter(node.outputs[0].outputs), None)
        if not sub_node or sub_node.op != "Sub":
            continue

        # 第二步 Sub -> Pow
        pow_node = next(iter(sub_node.outputs[0].outputs), None)
        if not pow_node or pow_node.op != "Pow":
            continue

        # 第三步 Pow -> ReduceMean
        reduce2_node = next(iter(pow_node.outputs[0].outputs), None)
        if not reduce2_node or reduce2_node.op != "ReduceMean":
            continue

        # 第四步 ReduceMean -> Add
        add_eps_node = next(iter(reduce2_node.outputs[0].outputs), None)
        if not add_eps_node or add_eps_node.op != "Add":
            continue

        # 第五步 Add -> Sqrt
        sqrt_node = next(iter(add_eps_node.outputs[0].outputs), None)
        if not sqrt_node or sqrt_node.op != "Sqrt":
            continue

        # 第六步 Sqrt -> Div
        div_node = next(iter(sqrt_node.outputs[0].outputs), None)
        if not div_node or div_node.op != "Div":
            continue

        # 第七步 Div -> Mul (乘gamma)
        mul_gamma_node = next(iter(div_node.outputs[0].outputs), None)
        if not mul_gamma_node or mul_gamma_node.op != "Mul":
            continue

        # 第八步 Mul -> Add (加beta)
        add_beta_node = next(iter(mul_gamma_node.outputs[0].outputs), None)
        if not add_beta_node or add_beta_node.op != "Add":
            continue

        # 匹配成功，保存这个子图
        patterns.append({
            "input": node.inputs[0],
            "output": add_beta_node.outputs[0],
            "nodes": [
                node, sub_node, pow_node, reduce2_node, add_eps_node,
                sqrt_node, div_node, mul_gamma_node, add_beta_node
            ]
        })

    return patterns

def replace_layernorm_patterns(graph, patterns):
    """用Custom节点替换掉找到的LayerNorm子图"""
    for pattern in patterns:
        input_tensor = pattern["input"]
        output_tensor = pattern["output"]

        custom_node = gs.Node(
            op="CustomLayerNorm",
            name="CustomLayerNorm_Replace",
            inputs=[input_tensor],
            outputs=[output_tensor]
        )
        graph.nodes.append(custom_node)

        # 删除老的子图节点
        for old_node in pattern["nodes"]:
            old_node.outputs.clear()

    graph.cleanup()

def main():
    # 1. 读入模型
    model = onnx.load("conv_layernorm_model.onnx")
    graph = gs.import_onnx(model)

    # 2. 查找所有LayerNorm子图
    layernorm_patterns = find_layernorm_patterns(graph)
    print(f"找到 {len(layernorm_patterns)} 个 LayerNorm 子图!")

    # 3. 替换为CustomLayerNorm节点
    replace_layernorm_patterns(graph, layernorm_patterns)

    # 4. 保存新模型
    new_model = gs.export_onnx(graph)
    onnx.save(new_model, "custom_layernorm.onnx")
    print("替换完成，新模型已保存为 custom_layernorm.onnx")

if __name__ == "__main__":
    main()
