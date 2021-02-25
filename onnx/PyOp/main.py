import onnx
import onnxruntime
from onnx import numpy_helper
import mymodule
import pdb
import numpy as np


if __name__=='__main__':
    model = './model.onnx'
    session = onnxruntime.InferenceSession(model, None)

    input_names = [meta.name for meta in session.get_inputs()]
    output_names = [meta.name for meta in session.get_outputs()]

    input_data = {}
    data = 0
    for name in input_names:
        data += 1
        input_data[name] = np.array([data]).astype('float32')
    

    result = session.run(output_names, input_data)
    print(result)
