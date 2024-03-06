// yolov9_ncnn.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "yolov9.h"

const int target_sizes =640;
const float norm_vals[3] ={1 / 255.f, 1 / 255.f , 1 / 255.f};
float prob_threshold = 0.10f;
float nms_threshold = 0.20f;

int main()
{
	cv::Mat m = cv::imread("./150.jpg", 1);
	//cv::Mat m_rgb;
	//cv::cvtColor(m, m_rgb, cv::COLOR_BGR2RGB);
	cv::Mat m1;
	cv::resize(m, m1, cv::Size(640, 640));


	std::vector<Object> objects;
	yolov9* yolo9 = new yolov9();
	yolo9->load(target_sizes, norm_vals);

	yolo9->detect(m, objects, prob_threshold, nms_threshold);

	yolo9->draw(m, objects);

	cv::imwrite("./final.jpg", m);

	return 0;
}


