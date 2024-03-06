#include "yolov9.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cpu.h"
#include<iostream>


//opencv计算两个bounding box的交集的像素点
static inline float intersection_area(const Object &a, const Object &b) {
	cv::Rect_<float> inter = a.rect & b.rect;
	return inter.area();
}

//二分排序
static void qsort_descent_inplace(std::vector<Object> &bobjects, int left, int right) {
	int i = left;
	int j = right;
	float p = bobjects[(left + right) / 2].prob;

	while (i <= j) {
		while (bobjects[i].prob > p)
			i++;

		while (bobjects[j].prob < p)
			j--;

		if (i <= j) {
			// swap:交换了两个容器的地址
			std::swap(bobjects[i], bobjects[j]);

			i++;
			j--;
		}
	}
	if (left < j) qsort_descent_inplace(bobjects, left, j);
	if (i < right) qsort_descent_inplace(bobjects, i, right);
}

static void qsort_descent_inplace(std::vector<Object> &objects) {
	if (objects.empty())
		return;

	qsort_descent_inplace(objects, 0, objects.size() - 1);
}

//nms 的实现
static void nms_sorted_bboxes(const std::vector<Object> &bobjects, std::vector<int> &picked,
	float nms_threshold) {
	picked.clear();

	const int n = bobjects.size();

	std::vector<float> areas(n);
	for (int i = 0; i < n; i++) {
		areas[i] = bobjects[i].rect.area();
	}

	for (int i = 0; i < n; i++) {
		const Object &a = bobjects[i];

		int keep = 1;
		for (int j = 0; j < (int)picked.size(); j++) {
			const Object &b = bobjects[picked[j]];

			// intersection over union  IoU计算
			float inter_area = intersection_area(a, b);
			float union_area = areas[i] + areas[picked[j]] - inter_area;
			// float IoU = inter_area / union_area
			if (inter_area / union_area > nms_threshold)
				keep = 0;
		}

		if (keep)
			picked.push_back(i);
	}
}

// 解析output
static void generate_proposals(const ncnn::Mat &feat_blob, float prob_threshold, std::vector<Object> &objects) {
	const int num_class = 1;
	const int feat_offset = num_class + 4;

	std::cout << feat_blob.c << ", " << feat_blob.h <<", " << feat_blob.w << std::endl;
	
	for (int i = 0; i < 8400; i++) {
		float score = feat_blob.row(4)[i];
		if (score >= prob_threshold) {
			Object obj;
			obj.prob = score;
			obj.label = 0;
			obj.rect.width = feat_blob.row(2)[i];
			obj.rect.height = feat_blob.row(3)[i];
			obj.rect.x = feat_blob.row(0)[i] - obj.rect.width / 2; //转坐上角点坐标
			obj.rect.y = feat_blob.row(1)[i] - obj.rect.height / 2;
			objects.push_back(obj);
			

		}
	}

}



yolov9::yolov9() {
	blob_pool_allocator.set_size_compare_ratio(0.f);
	workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int yolov9::load(int _target_size, const float *_norm_vals, bool use_gpu) {
	yolo.clear();
	blob_pool_allocator.clear();
	workspace_pool_allocator.clear();

	ncnn::set_cpu_powersave(2);
	ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

	yolo.opt = ncnn::Option();

//#if NCNN_VULKAN
//	yolo.opt.use_vulkan_compute = use_gpu;
//#endif

	yolo.opt.num_threads = ncnn::get_big_cpu_count();
	yolo.opt.blob_allocator = &blob_pool_allocator;
	yolo.opt.workspace_allocator = &workspace_pool_allocator;
	yolo.opt.use_vulkan_compute = false;
	//if int8
	//yolo.opt.use_int8_inference = true;


	//char parampath[256];
	//char modelpath[256];
	//sprintf(parampath, "%s.param", modeltype);
	//sprintf(modelpath, "%s.bin", modeltype);

	yolo.load_param("./models/yolov9-c.param");
	yolo.load_model("./models/yolov9-c.bin");

	target_size = _target_size;
	norm_vals[0] = _norm_vals[0];
	norm_vals[1] = _norm_vals[1];
	norm_vals[2] = _norm_vals[2];

	return 0;
}

int yolov9::detect(const cv::Mat &rgb, std::vector<Object> &objects, float prob_threshold,
	float nms_threshold) {

	// 前处理和yolov5保持一致 <--------------
	int img_w = rgb.cols;
	int img_h = rgb.rows;
	// letterbox pad to multiple of 32
	int w = img_w;
	int h = img_h;

	float scale = 1.f;
	if (w > h) {
		scale = (float)target_size / w;
		w = target_size;
		h = h * scale;
	}
	else {
		scale = (float)target_size / h;
		h = target_size;
		w = w * scale;
	}
	const int max_stride = 32;
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, img_w, img_h, w, h);

	int dw = target_size - w;
	int dh = target_size - h;
	dw = dw / 2;
	dh = dh / 2;

	// pad to target_size rectangle left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
	int top = static_cast<int>(std::round(dh - 0.1));
	int bottom = static_cast<int>(std::round(dh + 0.1));
	int left = static_cast<int>(std::round(dw - 0.1));
	int right = static_cast<int>(std::round(dw + 0.1));

	ncnn::Mat in_pad;
	ncnn::copy_make_border(in, in_pad, top, bottom, left, right, ncnn::BORDER_CONSTANT, 114.f);

	const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
	in_pad.substract_mean_normalize(0, norm_vals);

	//// pad to target_size rectangle
	//int wpad = (w + max_stride - 1) / max_stride * max_stride - w;
	//int hpad = (h + max_stride - 1) / max_stride * max_stride - h;
	//ncnn::Mat in_pad;
	//ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2,
	//	ncnn::BORDER_CONSTANT, 114.f);
	//in_pad.substract_mean_normalize(0, norm_vals);  //norm_vals: 1/255.0

	// 推理<----------------
	ncnn::Extractor ex = yolo.create_extractor();
	ex.input("images", in_pad);

	//解析推理结果 <-----------------
	std::vector<Object> boxes;
	ncnn::Mat out;
	ex.extract("1746", out);  //1x5x8400
	//CHW->CWH  (1,84,8400)->(1,8400,84)
	//ncnn::Mat out_t = transpose(out);

	generate_proposals(out, prob_threshold, boxes);

	// sort all proposals by score from highest to lowest
	qsort_descent_inplace(boxes);

	// apply nms with nms_threshold
	std::vector<int> picked;
	nms_sorted_bboxes(boxes, picked, nms_threshold);

	int count = picked.size();
	objects.resize(count);
	for (int i = 0; i < count; i++) {
		objects[i] = boxes[picked[i]];

		// adjust offset to original unpadded
		//std::cout << scale << ", " <<  wpad  << ", " << hpad  << std::endl;
		float x0 = (objects[i].rect.x - dw) / scale;
		float y0 = (objects[i].rect.y - dh) / scale;
		float x1 = (objects[i].rect.x + objects[i].rect.width - dw) / scale;
		float y1 = (objects[i].rect.y + objects[i].rect.height - dh) / scale;

		// clip
		x0 = (std::max)((std::min)(x0, (float)(img_w - 1)), 0.f);
		y0 = (std::max)((std::min)(y0, (float)(img_h - 1)), 0.f);
		x1 = (std::max)((std::min)(x1, (float)(img_w - 1)), 0.f);
		y1 = (std::max)((std::min)(y1, (float)(img_h - 1)), 0.f);

		objects[i].rect.x = x0;
		objects[i].rect.y = y0;
		objects[i].rect.width = x1 - x0;
		objects[i].rect.height = y1 - y0;

	}

	return 0;

}

int yolov9::draw(cv::Mat &rgb, const std::vector<Object> &objects) {
	static const char *class_names[] = {"pneumonia"};
	static const unsigned char colors[5][3] = {
			{54,  67,  244},
			{99,  30,  233},
			{176, 39,  156},
			{183, 58,  103},
			{181, 81,  63}
	};

	int color_index = 0;
	std::cout << objects.size() << std::endl;

	for (size_t i = 0; i < objects.size(); i++) {
		const Object &obj = objects[i];

		const unsigned char *color = colors[color_index % 19];
		color_index++;

		cv::Scalar cc(color[0], color[1], color[2]);

		cv::rectangle(rgb, obj.rect, cc, 2);

		char text[256];
		sprintf_s(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

		int baseLine = 0;
		cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

		int x = obj.rect.x;
		int y = obj.rect.y - label_size.height - baseLine;
		if (y < 0)
			y = 0;
		if (x + label_size.width > rgb.cols)
			x = rgb.cols - label_size.width;
		cv::rectangle(rgb, cv::Rect(cv::Point(x, y),
			cv::Size(label_size.width, label_size.height + baseLine)), cc,
			-1);
		cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? cv::Scalar(0, 0, 0)
			: cv::Scalar(255, 255, 255);
		cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5,
			textcc, 1);
	}

	return 0;
}