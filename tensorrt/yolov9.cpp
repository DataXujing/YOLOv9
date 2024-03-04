#include "yolov9.h"
#include <fstream>

yolov9::yolov9() {

}

IExecutionContext* yolov9::load_engine(std::string enginePath) {
	IExecutionContext* engine_context = new IExecutionContext;
	Logger gLogger;
	////初始化插件，调用plugin必须初始化plugin respo
	nvinfer1:initLibNvInferPlugins(&gLogger, "");

	nvinfer1::IRuntime* engine_runtime = nvinfer1::createInferRuntime(gLogger);
	std::string engine_filepath = enginePath;

	std::ifstream file;
	file.open(engine_filepath, std::ios::binary | std::ios::in);
	file.seekg(0, std::ios::end);
	int length = file.tellg();
	file.seekg(0, std::ios::beg);

	std::shared_ptr<char> data(new char[length], std::default_delete<char[]>());
	file.read(data.get(), length);
	file.close();

	//nvinfer1::ICudaEngine* engine_infer = engine_runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
	ICudaEngine* engine = engine_runtime->deserializeCudaEngine(data.get(), length, nullptr);

	engine_context = engine->createExecutionContext();

	return engine_context;
}

//和yolov5相同
void yolov9::preprocess(cv::Mat &img, float data[]) {
	int w, h, x, y;
	float r_w = INPUT_W / (img.cols*1.0);
	float r_h = INPUT_H / (img.rows*1.0);
	if (r_h > r_w) {
		w = INPUT_W;
		h = r_w * img.rows;
		x = 0;
		y = (INPUT_H - h) / 2;
	}
	else {
		w = r_h * img.cols;
		h = INPUT_H;
		x = (INPUT_W - w) / 2;
		y = 0;
	}
	cv::Mat re(h, w, CV_8UC3);
	cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
	//cudaResize(img, re);
	cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
	re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

	int i = 0;
	for (int row = 0; row < INPUT_H; ++row) {
		uchar* uc_pixel = out.data + row * out.step;
		for (int col = 0; col < INPUT_W; ++col) {
			data[i] = (float)uc_pixel[2] / 255.0;
			data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
			data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
			uc_pixel += 3;
			++i;
		}
	}
}

std::vector<Bbox> yolov9::postprocess(std::vector<Bbox> &out, int width, int height) {
	float gain = 640.0 / std::max(width, height);
	float pad_x = (640.0 - width * gain) / 2;
	float pad_y = (640.0 - height * gain) / 2;

	std::vector<Bbox> boxs;
	Bbox box;
	for (int i = 0; i < (int)out.size(); i++) {
		box.x1 = (out[i].x1 - pad_x) / gain;
		box.y1 = (out[i].y1 - pad_y) / gain;
		box.x2 = (out[i].x2 - pad_x) / gain;;
		box.y2 = (out[i].y2 - pad_y) / gain;;
		box.score = out[i].score;
		box.classes = out[i].classes;

		boxs.push_back(box);
	}

	return boxs;

}

cv::Mat yolov9::renderBoundingBox(cv::Mat image, const std::vector<Bbox> &bboxes) {
	for (const auto &rect : bboxes)
	{

		cv::Rect rst(rect.x1 ,rect.y1, rect.x2-rect.x1, rect.y2-rect.y1);
		cv::rectangle(image, rst, cv::Scalar(255, 204, 0), 2, cv::LINE_8, 0);
		//cv::rectangle(image, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2), cv::Point(rect.x + rect.w / 2, rect.y + rect.h / 2), cv::Scalar(255, 204,0), 3);

		int baseLine;
		std::string label = class_names[rect.classes] + ": " + std::to_string(rect.score * 100).substr(0, 4) + "%";
		cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_TRIPLEX, 0.5, 1, &baseLine);
		//int newY = std::max(rect.y, labelSize.height);
		rectangle(image, cv::Point(rect.x1, rect.y1 - round(1.5*labelSize.height)),
			cv::Point(rect.x1+ round(1.0*labelSize.width), rect.y1 + baseLine), cv::Scalar(255, 204, 0), cv::FILLED);
		cv::putText(image, label, cv::Point(rect.x1, rect.y1), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 204, 255));


	}
	return image;
}

