import onnx
from onnx import *

ad1_node = helper.make_node('Add', ['A','B'], ['S'])
mul_node = helper.make_node('Mul', ['C','D'], ['P'])
py1_node = helper.make_node(op_type = 'PyOp', #required, must be 'PyOp'
                            inputs = ['S','P'], #required
                            outputs = ['L','M','N'], #required
                            domain = 'pyopmulti_1', #required, must be unique
                            input_types = [TensorProto.FLOAT, TensorProto.FLOAT], #required
                            output_types = [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT], #required
                            module = 'mymodule', #required
                            class_name = 'Multi_1', #required
                            compute = 'compute', #optional, 'compute' by default
                            W1 = '5', W2 = '7', W3 = '9') #optional, must all be strings
ad2_node = helper.make_node('Add', ['L','M'], ['H'])
py2_node = helper.make_node('PyOp',['H','N','E'],['O','W'], domain = 'pyopmulti_2',
                            input_types = [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT],
                            output_types = [TensorProto.FLOAT, TensorProto.FLOAT],
                            module = 'mymodule', class_name = 'Multi_2')
sub_node = helper.make_node('Sub', ['O','W'], ['F'])

opset = [
        helper.make_operatorsetid('pyopmulti_1',13),
        helper.make_operatorsetid('pyopmulti_2',13),
        ]

inputs = [
         helper.make_tensor_value_info('A', TensorProto.FLOAT, shape=(1,)),
         helper.make_tensor_value_info('B', TensorProto.FLOAT, shape=(1,)),
         helper.make_tensor_value_info('C', TensorProto.FLOAT, shape=(1,)),
         helper.make_tensor_value_info('D', TensorProto.FLOAT, shape=(1,)),
         helper.make_tensor_value_info('E', TensorProto.FLOAT, shape=(1,)),
        ]

outputs = [helper.make_tensor_value_info('F',TensorProto.FLOAT, shape=(1,))]

graph = helper.make_graph([ad1_node,mul_node,py1_node,ad2_node,py2_node,sub_node], 'multi_pyop_graph', inputs, None)
model = helper.make_model(graph, producer_name = 'pyop_model', opset_imports = opset)
onnx.save(model, './model.onnx')
