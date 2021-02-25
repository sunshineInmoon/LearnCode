import onnx

model = onnx.load('mnist/model.onnx')
graph = model.graph


for w in graph.value_info:
    graph.output.extend([w])

print(graph.output)

onnx.save(model, 'model.onnx')

