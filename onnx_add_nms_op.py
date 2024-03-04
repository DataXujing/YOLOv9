
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


def get_nms_input(graph,class_num=1,output_name="Concat_1278"):
	# input
	# Concat_1278时DualDDetection输出的的第二个输出，这个输出时main branch的输出
	# 不同于之前的YOLO,该部分已经包含了ancher的映射和处理，直接解析就好了
	origin_output = [node for node in graph.nodes if node.name == output_name][0]  
	print(origin_output.outputs)

	# tanspose 1x8400x5
	output_2 = gs.Variable(name="output_reshape",shape=(1,8400,4+class_num),dtype=np.float32)
	output_2_node = gs.Node(op="Transpose",inputs=[origin_output.outputs[0]],outputs=[output_2],attrs={"perm":[0,2,1]})

	# box_xywh,box_score
	box_xywh = gs.Variable(name="box_input",shape=(1,8400,4),dtype=np.float32)
	box_score = gs.Variable(name="score_input",shape=(1,8400,class_num),dtype=np.float32)

	starts_wh = gs.Constant("starts_wh",values=np.array([0,0,0],dtype=np.int64))
	ends_wh = gs.Constant("ends_wh",values=np.array([1,8400,4],dtype=np.int64))

	starts_conf = gs.Constant("starts_conf",values=np.array([0,0,4],dtype=np.int64))
	ends_conf = gs.Constant("ends_conf",values=np.array([1,8400,4+class_num],dtype=np.int64))

	box_xywh_node = gs.Node(op="Slice",inputs=[output_2,starts_wh,ends_wh],outputs=[box_xywh])
	box_score_node = gs.Node(op="Slice",inputs=[output_2,starts_conf,ends_conf],outputs=[box_score])


	graph.nodes.extend([output_2_node,box_xywh_node,box_score_node,])
	graph.outputs = [ box_xywh,box_score ]

	graph.cleanup().toposort()
	# onnx.save(gs.export_onnx(graph),"./last_1.onnx")

	return graph


# graph中插入EfficientNMS plugin op
def create_and_add_plugin_node(graph, max_output_boxes):
    
    batch_size = graph.inputs[0].shape[0]
    print("The batch size is: ", batch_size)
    input_h = graph.inputs[0].shape[2]
    input_w = graph.inputs[0].shape[3]

    tensors = graph.tensors()
    boxes_tensor = tensors["box_input"]  
    confs_tensor = tensors["score_input"]

    num_detections = gs.Variable(name="num_detections").to_variable(dtype=np.int32, shape=[batch_size, 1])
    nmsed_boxes = gs.Variable(name="detection_boxes").to_variable(dtype=np.float32, shape=[batch_size, max_output_boxes, 4])
    nmsed_scores = gs.Variable(name="detection_scores").to_variable(dtype=np.float32, shape=[batch_size, max_output_boxes])
    nmsed_classes = gs.Variable(name="detection_classes").to_variable(dtype=np.int32, shape=[batch_size, max_output_boxes])

    new_outputs = [num_detections, nmsed_boxes, nmsed_scores, nmsed_classes]

    mns_node = gs.Node(
        op="EfficientNMS_TRT",
        attrs=create_attrs(max_output_boxes),
        inputs=[boxes_tensor, confs_tensor],
        outputs=new_outputs)

    graph.nodes.append(mns_node)
    graph.outputs = new_outputs

    return graph.cleanup().toposort()


def create_attrs(max_output_boxes=100):

    attrs = {}

    attrs["score_threshold"] = 0.25 
    attrs["iou_threshold"] = 0.45  
    attrs["max_output_boxes"] = max_output_boxes
    attrs["background_class"] = -1
    attrs["score_activation"] = False
    attrs["class_agnostic"] = True
    attrs["box_coding"] = 1
    # 001 is the default plugin version the parser will search for, and therefore can be omitted,
    # but we include it here for illustrative purposes.
    attrs["plugin_version"] = "1"

    return attrs


if __name__ == "__main__":
	onnx_path = "./runs/train/exp/weights/last.onnx"
	graph = gs.import_onnx(onnx.load(onnx_path))

	# 添加op得到Efficient NMS plugin的input
	graph = get_nms_input(graph,class_num=1,output_name="Concat_1278")

	# 添加Efficient NMS plugin
	graph = create_and_add_plugin_node(graph, 100)

	# 保存图结构
	onnx.save(gs.export_onnx(graph),"./runs/train/exp/weights/last_nms.onnx")






