# -*- coding:utf-8 -*-
import onnx
from onnx import helper
from onnx import numpy_helper
import sys
import getopt
import numpy as np
import pdb


# 加载模型
def load_onnx_model(path):
    model = onnx.load(path)
    return model


# 获取节点和节点的输入输出名列表，一般节点的输入将来自于上一层的输出放在列表前面，
# 参数放在列表后面
def get_node_and_io(node_name, model):
    for i in range(len(model.graph.node)):
        if model.graph.node[i].name == node_name:
            Node = model.graph.node[i]
            input_name = model.graph.node[i].input
            output_name = model.graph.node[i].output
    return Node, input_name, output_name


# 获取对应输入信息
def get_input_tensor_value_info(input_name, model):
    in_tvi = []
    for name in input_name:
        # pdb.set_trace()
        for params_input in model.graph.input:
            if params_input.name == name:
                in_tvi.append(params_input)
        for inner_output in model.graph.value_info:
            if inner_output.name == name:
                in_tvi.append(inner_output)
    return in_tvi


# 获取对应输出信息
def get_output_tensor_value_info(output_name, model):
    out_tvi = []
    for name in output_name:
        for params_out in model.graph.output:
            if params_out.name == name:
                out_tvi.append(params_out)
        for inner_output in model.graph.value_info:
            if inner_output.name == name:
                out_tvi.append(inner_output)
    return out_tvi


# 获取对应超参数信息
def get_init_tensor_value(input_name, model):
    init_t = []
    for name in input_name:
        for init in model.graph.initializer:
            # pdb.set_trace()
            if init.name == name:
                init_t.append(init)
    return init_t


# 获取超参数的值
def get_init_value(input_name, model):
    val = []
    for name in input_name:
        for init in model.graph.initializer:
            if init.name == name:
                shape = init.dims
                pdb.set_trace()
                data = init.float_data
                val.append(np.array(data).reshape(shape))
    return val


# 获取节点数量, 不会包括输入和输出节点
def get_node_num(model):
    return len(model.graph.node)


# 获取节点类型
def get_node_type(model):
    op_name = []
    for i in range(len(model.graph.node)):
        if model.graph.node[i].op_type not in op_name:
            op_name.append(model.graph.node[i].op_type)
    return op_name


# 获取节点名列表
def get_node_name_list(model):
    node_list = []
    graph = model.graph
    for node in graph.node:
        node_list.append(node.name)
    return node_list


# 获取输入信息
def get_model_input_info(model):
    return model.graph.input


# 获取输出信息
def get_model_output_info(model):
    return model.graph.output


if __name__ == '__main__':
    model = load_onnx_model('mnist_hua_remove_inputs.onnx')
    # pdb.set_trace()
    Node, input_name, output_name = get_node_and_io('Convolution28', model)
    # print(Node, input_name, output_name)
    in_tvi = get_input_tensor_value_info(['Input3'], model)
    # print(in_tvi)
    out_tvi = get_output_tensor_value_info(['Plus214_Output_0'], model)
    # print(out_tvi)
    init_t = get_init_tensor_value(['Parameter5'], model)
    # print(init_t)
    node_num = get_node_num(model)
    # print(node_num)
    op_name = get_node_type(model)
    # print(op_name)
    node_name_list = get_node_name_list(model)
    # print(node_name_list)
    input_info = get_model_input_info(model)
    # print(input_info)
    output_info = get_model_output_info(model)
    # print(output_info)
    val = get_init_value(['Parameter5', 'Parameter6'], model)
    print(val)