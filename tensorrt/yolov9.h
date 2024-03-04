#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>

#include "cuda_runtime_api.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "logging.h"


#define BATCH_SIZE 1
#define INPUT_W 640
#define INPUT_H 640
#define INPUT_SIZE 640

using namespace nvinfer1;
using namespace sample;


// box x1,y1,x2,y2
struct Bbox {
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int classes;
};


class yolov9
{
public:
	
	yolov9();

	IExecutionContext* load_engine(std::string enginePath);

	void preprocess(cv::Mat &img, float data[]);

	std::vector<Bbox> postprocess(std::vector<Bbox> &out, int width, int height);

	cv::Mat renderBoundingBox(cv::Mat image, const std::vector<Bbox> &bboxes);

public:
	//ICudaEngine* engine;
	//IExecutionContext* engine_context;
	cv::Mat image;
	std::vector<std::string> class_names = { "pneumonia" };



};

