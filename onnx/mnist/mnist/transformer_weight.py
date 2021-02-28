# -*- coding:utf-8 -*-
import onnx
from onnx import helper
from onnx import numpy_helper
import numpy as np
import pdb


# 获取init名字
def get_init_names(model):
    graph = model.graph
    init_names = []
    for init in graph.initializer:
        init_names.append(init.name)
    return init_names


# 获取权重并转换成numpy
def get_init_value_to_numpy(input_name, model):
    graph = model.graph
    params = dict()
    for name in input_name:
        for init in graph.initializer:
            if name == init.name:
                # pdb.set_trace()
                shape = init.dims
                data_type = onnx.TensorProto.DataType.Name(init.data_type).lower()
                if data_type == 'int64':
                    data = init.int64_data
                elif data_type == 'int32':
                    data = init.int32_data
                elif data_type == 'float':
                    data = init.float_data
                if not data:
                    pdb.set_trace()
                param = np.array(data).reshape(shape)
                if name not in params.keys():
                    params[name] = param
                else:
                    print('ERROR: {}'.format(name))
    return params


# 获取权重TensorProto
def get_init(input_name, model):
    graph = model.graph
    params = dict()
    for name in input_name:
        for init in graph.initializer:
            if name == init.name:
                params[name] = init
    return params


# 重新设置model参数
def set_model_params(init_val, model, new_model_path):
    graph = model.graph
    init_num = len(graph.initializer)
    for i in range(init_num):
        init = graph.initializer[i]
        pdb.set_trace()
        name = init.name
        init_data = numpy_helper.to_array(init)
        data_type = init_data.dtype
        # data = init_val[name]
        data = np.zeros(init_data.shape, dtype=data_type)
        """
        data_type = onnx.TensorProto.DataType.Name(init.data_type).lower()
        if data_type == 'int64':
            init.int64_data = data.flatten()
        elif data_type == 'int32':
            init.int32_data = data.flatten()
        elif data_type == 'float':
            init.float_data = data.flatten()
        """
        # init = numpy_helper.from_array(data, name=name)
        tensor = onnx.TensorProto()
        tensor.dims.extend(data.shape)
        tensor.name = name
        tensor.float_data.extend(data.flatten().tolist())
        tensor.data_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[data.dtype]
        graph.initializer[i] = tensor
    onnx.save(model, new_model_path)


# 重新设置model参数
def set_model_init(init_val, model, new_model_path):
    graph = model.graph
    for i in range(len(graph.initializer)):
        name = model.graph.initializer[i].name
        # pdb.set_trace()
        new_init = init_val[name]
        old_init = model.graph.initializer[i]
        model.graph.initializer.remove(old_init)
        model.graph.initializer.insert(i, new_init)
    onnx.save(model, new_model_path)


if __name__ == '__main__':
    model = onnx.load('model.onnx')
    init_names = get_init_names(model)
    # print(init_names)
    init_val = get_init(init_names, model)
    # print(init_val)

    model_hua = onnx.load('mnist_hua_remove_inputs.onnx')
    # set_model_init(init_val, model_hua, 'mnist_new_params.onnx')
    set_model_params(init_val, model_hua, 'mnist_new_params.onnx')