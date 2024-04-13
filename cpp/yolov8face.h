# ifndef YOLOV8FACE
# define YOLOV8FACE
#include <fstream>
#include <sstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>  ///如果使用cuda加速，需要取消注释
#include <onnxruntime_cxx_api.h>
#include"utils.h"


class Yolov8Face
{
public:
	Yolov8Face(std::string modelpath, const float conf_thres=0.5, const float iou_thresh=0.4);
	void detect(cv::Mat srcimg, std::vector<Bbox> &boxes);   ////只返回检测框,置信度和5个关键点这两个信息在后续的模块里没有用到
private:
	void preprocess(cv::Mat img);
	std::vector<float> input_image;
	int input_height;
	int input_width;
	float ratio_height;
	float ratio_width;
	float conf_threshold;
	float iou_threshold;

	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Face Detect");
	Ort::Session *ort_session = nullptr;
	Ort::SessionOptions sessionOptions = Ort::SessionOptions();
	std::vector<char*> input_names;
	std::vector<char*> output_names;
	std::vector<std::vector<int64_t>> input_node_dims; // >=1 outputs
	std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs
	Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
};
#endif