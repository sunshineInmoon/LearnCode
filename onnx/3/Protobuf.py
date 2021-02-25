import onnx
from onnx import helper
import pdb

graph_proto = helper.make_graph(
        [
            helper.make_node('FC', ['X','W1','B1'], ['H1']),
            helper.make_node('Relu', ['H1'], ['R1']),
            helper.make_node('FC',['R1','W2','R2'],['Y']),
        ],
        'MLP',
        [
            helper.make_tensor_value_info('X',onnx.TensorProto.FLOAT,[1]),
            helper.make_tensor_value_info('W1',onnx.TensorProto.FLOAT,[1]),
            helper.make_tensor_value_info('B1',onnx.TensorProto.FLOAT,[1]),
            helper.make_tensor_value_info('W2',onnx.TensorProto.FLOAT,[1]),
            helper.make_tensor_value_info('B2',onnx.TensorProto.FLOAT,[1]),
        ],
        [
            helper.make_tensor_value_info('Y',onnx.TensorProto.FLOAT,[1]),
        ]
)

node_proto = helper.make_node(
    "graph",
    ["Input", "W1","B1","W2","B2"],
    ["Output"],
    graph=[graph_proto],
)

pdb.set_trace()
