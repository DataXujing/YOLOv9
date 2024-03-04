// yolov9_trt.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv.hpp>

#include "cuda_runtime_api.h"
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"

# include "yolov9.h"


float h_input[INPUT_SIZE * INPUT_SIZE * 3];
int h_output_0[1];   //1
float h_output_1[1 * 100 * 4];   //1
float h_output_2[1 * 100];   //1
float h_output_3[1 * 100];   //1

int main()
{
	yolov9 *yolo = new yolov9;

	IExecutionContext* engine_context = yolo->load_engine("./model/yolov9-c.plan");

	if (engine_context == nullptr)
	{
		std::cerr << "failed to create tensorrt execution context." << std::endl;
	}


	//cv2读图片
	cv::Mat image;
	image = cv::imread("./150.jpg", 1);

	yolo->preprocess(image, h_input);

	void* buffers[5];
	cudaMalloc(&buffers[0], INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float));  //<- input
	cudaMalloc(&buffers[1], 1 * sizeof(int)); //<- num_detections
	cudaMalloc(&buffers[2], 1 * 100 * 4 * sizeof(float)); //<- nmsed_boxes
	cudaMalloc(&buffers[3], 1 * 100 * sizeof(float)); //<- nmsed_scores
	cudaMalloc(&buffers[4], 1 * 100 * sizeof(int)); //<- nmsed_classes

	cudaMemcpy(buffers[0], h_input, INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float), cudaMemcpyHostToDevice);

	// -- do execute --------//
	engine_context->executeV2(buffers);

	cudaMemcpy(h_output_0, buffers[1], 1 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_output_1, buffers[2], 1 * 100 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_output_2, buffers[3], 1 * 100 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_output_3, buffers[4], 1 * 100 * sizeof(int), cudaMemcpyDeviceToHost);

	std::cout << h_output_0[0] << std::endl;
	std::vector<Bbox> pred_box;
	for (int i = 0; i < h_output_0[0]; i++) {
		Bbox box;
		box.x1 = h_output_1[i * 4];
		box.y1 = h_output_1[i * 4 + 1];
		box.x2 = h_output_1[i * 4 + 2];
		box.y2 = h_output_1[i * 4 + 3];
		box.score = h_output_2[i];
		box.classes = h_output_3[i];

		std::cout << box.classes << "," << box.score << std::endl;

		pred_box.push_back(box);
	}

	std::vector<Bbox> out = yolo->postprocess(pred_box, image.cols, image.rows);
	cv::Mat img = yolo->renderBoundingBox(image, out);

	cv::imwrite("final.jpg", img);

	cv::namedWindow("Image", 1);//创建窗口
	cv::imshow("Image", img);//显示图像

	cv::waitKey(0); //等待按键

	cudaFree(buffers[0]);
	cudaFree(buffers[1]);
	cudaFree(buffers[2]);
	cudaFree(buffers[3]);
	cudaFree(buffers[4]);

	delete engine_context;

}


