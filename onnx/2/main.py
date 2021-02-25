import torch.onnx
#help(torch.onnx.export)
import torchvision
import torch 

dummy_input = torch.randn(1,3,224,224)
#model = torchvision.models.alexnet(pretrained=True)
#torch.onnx.export(model, dummy_input, "alexnet.onnx")

import onnx

model = onnx.load('alexnet.onnx')
onnx.checker.check_model(model)

#print(onnx.helper.printable_graph(model.graph))

import pdb
pdb.set_trace()
