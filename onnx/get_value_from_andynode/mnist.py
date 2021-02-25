import json
import sys
import os
import time
import numpy as np
import cv2
import onnx
import onnxruntime
from onnx import numpy_helper
import pdb

if __name__=='__main__':

    #model_dir = './mnist'
    #model=model_dir+'/model.onnx'
    model = 'model.onnx'
    path=sys.argv[1]
    
    pdb.set_trace()
    #Preprocess the image
    img = cv2.imread(path)
    img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    img = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_AREA)
    img.resize((1, 1, 28, 28))
    img = img.astype('float32')
    
    #data = json.dumps({'data': img.tolist()})
    #data = np.array(json.loads(data)['data']).astype('float32')
    
    # ready to pass the data to the model for inference
    session = onnxruntime.InferenceSession(model, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output_names = [meta.name for meta in session.get_outputs()]
    #print(input_name)
    #print(output_name)
    print(len(output_names))
    print(output_names)
    
    result = session.run(output_names[:11], {input_name: img})
    print(len(result))
    #prediction = int(np.argmax(np.array(result).squeeze(),axis=0))
    #print(prediction)
