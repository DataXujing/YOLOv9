
'''

xujing

2024-3-03

注意DualDDetection export的时候有两个output

yolov9-c.pt and yolov9-e.pt return list 
output[0] is prediction of aux branch, output[1] is prediction of main branch.

https://github.com/WongKinYiu/yolov9/issues/130#issuecomment-1974792028

本代码将编辑onnx图结构增加EfficentNMS plugin，实现端到端的YOLOv9

'''

import onnx_graphsurgeon as gs
import numpy as np
import onnx


def get_ncnn_graph(graph,class_num=1,output_name="Concat_1186"):
	# input
	# Concat_1278时DualDDetection输出的的第二个输出，这个输出时main branch的输出
	# 不同于之前的YOLO,该部分已经包含了ancher的映射和处理，直接解析就好了
	origin_output = [node for node in graph.nodes if node.name == output_name][0]  
	print(origin_output.outputs)

	graph.outputs = [origin_output.outputs[0]]

	graph.cleanup().toposort()
	# onnx.save(gs.export_onnx(graph),"./last_1.onnx")

	return graph


if __name__ == "__main__":
	# onnx_path = "./runs/train/exp/weights/last.onnx"
	onnx_path = "./pretrain/yolov9-c.onnx"
	graph = gs.import_onnx(onnx.load(onnx_path))
 
	graph = get_ncnn_graph(graph,class_num=1,output_name="Concat_1184")

	# 保存图结构
	# onnx.save(gs.export_onnx(graph),"./runs/train/exp/weights/last_ncnn.onnx")
	onnx.save(gs.export_onnx(graph),"./pretrain/last_ncnn.onnx")






