"""Emit trtexec constraints for dynamic RoPE arithmetic (keep coordinates FP32)."""
import sys
import onnx


def constraints(path):
    graph = onnx.load(path, load_external_data=False).graph
    types = {v.name: v.type.tensor_type.elem_type for v in graph.value_info}
    names = []
    for node in graph.node:
        if node.op_type == 'LayerNormalization':
            break
        if node.op_type in ('Range', 'Div', 'Mul', 'Sub', 'Cos', 'Sin'):
            if all(types.get(o) == onnx.TensorProto.FLOAT for o in node.output):
                names.append(node.name)
    if not any('cos' in n for n in names) or not any('sin' in n for n in names):
        raise RuntimeError('Could not identify the exported RoPE subgraph')
    return ','.join(f'{name}:fp32' for name in names)


if __name__ == '__main__':
    print(constraints(sys.argv[1]))
