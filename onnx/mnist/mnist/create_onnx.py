# -*- coding:utf-8 -*-
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from onnx import version_converter
import numpy as np


def create_onnx():
    inputs = []
    outputs = []
    node_list = []
    initializer_list = []

    # conv
    X = helper.make_tensor_value_info('Input3', TensorProto.FLOAT, [1,1,28,28])
    inputs.append(X)
    W = helper.make_tensor_value_info('Parameter5', TensorProto.FLOAT, [8,1,5,5])
    w = helper.make_tensor('Parameter5', TensorProto.FLOAT, [8,1,5,5], np.random.random((8,1,5,5)).flatten())
    inputs.append(W)
    initializer_list.append(w)
    Y = helper.make_tensor_value_info('Convolution28_Output_0', TensorProto.FLOAT, [1,8,28,28])
    conv1_node = helper.make_node(
            'Conv', ['Input3', 'Parameter5'], ['Convolution28_Output_0'], 'Convolution28', 
            auto_pad='SAME_UPPER', dilations=[1,1], group=1, kernel_shape=[5,5], strides=[1,1])
    node_list.append(conv1_node)

    # add
    B = helper.make_tensor_value_info('Parameter6', TensorProto.FLOAT, [8,1,1])
    inputs.append(B)
    b = helper.make_tensor('Parameter6', TensorProto.FLOAT, [8,1,1], np.random.random((8,1,1)).flatten())
    initializer_list.append(b)
    Y = helper.make_tensor_value_info('Plus30_Output_0', TensorProto.FLOAT, [1,8,28,28])
    add1_node = helper.make_node(
            'Add',['Convolution28_Output_0', 'Parameter6'],['Plus30_Output_0'], 'Plus30', )
    node_list.append(add1_node)

    # relu
    Y = helper.make_tensor_value_info('ReLU32_Output_0', TensorProto.FLOAT, [1,8,28,28])
    relu1_node = helper.make_node(
            'Sigmoid',['Plus30_Output_0'],['ReLU32_Output_0'],'ReLU32',)
    node_list.append(relu1_node)
   
    # maxpool
    Y = helper.make_tensor_value_info('Pooling66_Output_0', TensorProto.FLOAT, [1,8,14,14])
    maxpool1_node = helper.make_node(
            'MaxPool', ['ReLU32_Output_0'], ['Pooling66_Output_0'], 'Pooling66', 
            auto_pad='NOTSET', kernel_shape=[2,2], pads=[0, 0, 0, 0], strides=[2, 2])
    node_list.append(maxpool1_node)

    # conv
    W = helper.make_tensor_value_info('Parameter87', TensorProto.FLOAT, [16,8,5,5])
    w = helper.make_tensor('Parameter87', TensorProto.FLOAT, [16,8,5,5], np.random.random((16,8,5,5)).flatten())
    inputs.append(W)
    initializer_list.append(w)
    Y = helper.make_tensor_value_info('Convolution110_Output_0', TensorProto.FLOAT, [1,16,14,14])
    conv1_node = helper.make_node(
            'Conv', ['Pooling66_Output_0', 'Parameter87'], ['Convolution110_Output_0'], 'Convolution110', 
            auto_pad='SAME_UPPER', dilations=[1,1], group=1, kernel_shape=[5,5], strides=[1,1])
    node_list.append(conv1_node)

    # add
    B = helper.make_tensor_value_info('Parameter88', TensorProto.FLOAT, [16,1,1])
    inputs.append(B)
    b = helper.make_tensor('Parameter88', TensorProto.FLOAT, [16,1,1], np.random.random((16,1,1)).flatten())
    initializer_list.append(b)
    Y = helper.make_tensor_value_info('Plus112_Output_0', TensorProto.FLOAT, [1,16,14,14])
    add1_node = helper.make_node(
            'Add',['Convolution110_Output_0', 'Parameter88'],['Plus112_Output_0'], 'Plus112', )
    node_list.append(add1_node)

    # relu
    Y = helper.make_tensor_value_info('ReLU114_Output_0', TensorProto.FLOAT, [1,16,14,14])
    relu1_node = helper.make_node(
            'Sigmoid',['Plus112_Output_0'],['ReLU114_Output_0'],'ReLU114', )
    node_list.append(relu1_node)
   
    # maxpool
    Y = helper.make_tensor_value_info('Pooling160_Output_0', TensorProto.FLOAT, [1,16,4,4])
    maxpool1_node = helper.make_node(
            'MaxPool', ['ReLU114_Output_0'], ['Pooling160_Output_0'], 'Pooling160', 
            auto_pad='NOTSET', kernel_shape=[3,3], pads=[0, 0, 0, 0], strides=[3, 3])
    node_list.append(maxpool1_node)
    
    # Reshape1
    Shape = helper.make_tensor_value_info('Pooling160_Output_0_reshape0_shape', TensorProto.INT64, [2])
    inputs.append(Shape)
    shape = helper.make_tensor('Pooling160_Output_0_reshape0_shape', TensorProto.INT64, [2], np.array([1,256]))
    initializer_list.append(shape)
    Y = helper.make_tensor_value_info('Pooling160_Output_0_reshape0', TensorProto.FLOAT, [1,256])
    reshape_node=helper.make_node(
            'Reshape', ['Pooling160_Output_0', 'Pooling160_Output_0_reshape0_shape'], ['Pooling160_Output_0_reshape0'], 'Times212_reshape0', )
    node_list.append(reshape_node)

    # Reshape2
    Data = helper.make_tensor_value_info('Parameter193', TensorProto.FLOAT, [16,4,4,10])
    inputs.append(Data)
    data = helper.make_tensor('Parameter193', TensorProto.FLOAT, [16,4,4,10], np.random.random((16,4,4,10)).flatten())
    initializer_list.append(data)
    Shape = helper.make_tensor_value_info('Parameter193_reshape1_shape', TensorProto.INT64, [2])
    inputs.append(Shape)
    shape = helper.make_tensor('Parameter193_reshape1_shape', TensorProto.INT64, [2], np.array([256, 10]))
    initializer_list.append(shape)
    Y = helper.make_tensor_value_info('Parameter193_reshape1', TensorProto.FLOAT, [256, 10])
    reshape_node = helper.make_node(
            'Reshape', ['Parameter193', 'Parameter193_reshape1_shape'], ['Parameter193_reshape1'], 'Times212_reshape1', )
    node_list.append(reshape_node)

    # matmul
    Y = helper.make_tensor_value_info('Times212_Output_0', TensorProto.FLOAT, [1,10])
    matmul_node = helper.make_node(
            'MatMul', ['Pooling160_Output_0_reshape0', 'Parameter193_reshape1'], ['Times212_Output_0'], 'Times212', )
    node_list.append(matmul_node)

    # add
    B = helper.make_tensor_value_info('Parameter194', TensorProto.FLOAT, [1,10])
    inputs.append(B)
    b = helper.make_tensor('Parameter194', TensorProto.FLOAT, [1,10], np.random.random((1,10)).flatten())
    initializer_list.append(b)
    Y = helper.make_tensor_value_info('Plus214_Output_0', TensorProto.FLOAT, [1,10])
    add1_node = helper.make_node(
            'Add',['Times212_Output_0', 'Parameter194'],['Plus214_Output_0'], 'Plus214', )
    node_list.append(add1_node)
    
    # create graph
    outputs.append(Y)
    graph_def = helper.make_graph(
            node_list,
            'mnist_hua',
            inputs,
            outputs,
            initializer_list,
            )


    #opset = [helper.make_operatorsetid('ai.onnx', 13)]
    # create model
    model_def = helper.make_model(
            graph_def,
            producer_name='onnx-example',)
            #opset_imports=opset)

    # converter version
    model_def = version_converter.convert_version(model_def, 13)
    # polish model
    model_def = onnx.utils.polish_model(model_def)
    # check model
    onnx.checker.check_model(model_def)
    # save model
    onnx.save(model_def, 'mnist_hua.onnx')


if __name__=='__main__':
    create_onnx()


