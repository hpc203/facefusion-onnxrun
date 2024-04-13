# ifndef FACEENHANCE
# define FACEENHANCE
#include <fstream>
#include <sstream>
#include "opencv2/opencv.hpp"
//#include <cuda_provider_factory.h>  ///如果使用cuda加速，需要取消注释
#include <onnxruntime_cxx_api.h>
#include"utils.h"


class FaceEnhance
{
public:
	FaceEnhance(std::string modelpath);
	cv::Mat process(cv::Mat target_img, const std::vector<cv::Point2f> target_landmark_5);
private:
	void preprocess(cv::Mat target_img, const std::vector<cv::Point2f> face_landmark_5, cv::Mat& affine_matrix, cv::Mat& box_mask);
	std::vector<float> input_image;
	int input_height;
	int input_width;
	std::vector<cv::Point2f> normed_template;
    const float FACE_MASK_BLUR = 0.3;
	const int FACE_MASK_PADDING[4] = {0, 0, 0, 0};

	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Face Enhance");
	Ort::Session *ort_session = nullptr;
	Ort::SessionOptions sessionOptions = Ort::SessionOptions();
	std::vector<char*> input_names;
	std::vector<char*> output_names;
	std::vector<std::vector<int64_t>> input_node_dims; // >=1 outputs
	std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs
	Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
};
#endif