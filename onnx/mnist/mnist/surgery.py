# -*- coding:utf-8 -*-
"""This a script for help
Author: Hua.Z
Date: 20210302
"""
import pdb
import onnx
from onnx import helper
from onnx import TensorProto
from onnx import shape_inference


ONNX_DTYPE = {
    0: TensorProto.FLOAT,
    1: TensorProto.FLOAT,
    2: TensorProto.UINT8,
    3: TensorProto.INT8,
    4: TensorProto.UINT16,
    5: TensorProto.INT16,
    6: TensorProto.INT32,
    7: TensorProto.INT64,
    8: TensorProto.STRING,
    9: TensorProto.BOOL
}


def add_extra_output_to_model(old_model_path, new_model_path):
    model = onnx.load(old_model_path)
    model.graph.output.extend(model.graph.value_info)
    onnx.save(model, new_model_path)


def insert_node(model_path, index, node, new_model_path):
    model = onnx.load(model_path)
    pdb.set_trace()
    model.graph.node.insert(index, node)
    model.graph.node[index + 1].input[0] = 'Inset_output'
    onnx.save(model, new_model_path)


def insert_node_V1(old_model_path, index, node_type, out_name, node_name, new_model_path):
    model = onnx.load(old_model_path)
    # inferred_model = shape_inference.infer_shapes(model)
    up_node_value_info = model.graph.value_info[index - 1]
    up_node_output_name = up_node_value_info.name
    up_node_output_type = ONNX_DTYPE[up_node_value_info.type.tensor_type.elem_type]
    up_node_output_dim = [d.dim_value for d in up_node_value_info.type.tensor_type.shape.dim]

    # Y = helper.make_tensor_value_info(up_node_output_name, up_node_output_type, up_node_output_dim)
    insert_node = helper.make_node(
         node_type, [up_node_output_name], [out_name], node_name,)

    model.graph.node.insert(index, insert_node)
    model.graph.node[index + 1].input[0] = out_name
    onnx.save(model, new_model_path)
    # pdb.set_trace()


if __name__ == '__main__':
    # add_extra_output_to_model('model.onnx', 'model_extra_output.onnx')
    # add_extra_output_to_model('mnist_hua.onnx', 'model_extra_output_v1.onnx')

    # Sigmoid
    # Y = helper.make_tensor_value_info('ReLU114_Output_0', TensorProto.FLOAT, [1, 16, 14, 14])
    # sigmoid_node = helper.make_node(
    #     'Sigmoid', ['ReLU114_Output_0'], ['Inset_output'], 'Insert_Node', )
    # insert_node('mnist_hua.onnx', 7, sigmoid_node, 'mnist_hua_insert.onnx')

    # Sigmoid V1
    insert_node_V1('mnist_hua.onnx', 7, 'Sigmoid', 'Insert_output', 'Insert_Node', 'mnist_hua_insert_V1.onnx')

