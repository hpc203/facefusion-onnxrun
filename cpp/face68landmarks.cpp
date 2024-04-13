#include "face68landmarks.h"

using namespace cv;
using namespace std;
using namespace Ort;

Face68Landmarks::Face68Landmarks(string model_path)
{
    /// OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);   ///如果使用cuda加速，需要取消注释

    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    /// std::wstring widestr = std::wstring(model_path.begin(), model_path.end());  ////windows写法
    /// ort_session = new Session(env, widestr.c_str(), sessionOptions); ////windows写法
    ort_session = new Session(env, model_path.c_str(), sessionOptions); ////linux写法

    size_t numInputNodes = ort_session->GetInputCount();
    size_t numOutputNodes = ort_session->GetOutputCount();
    AllocatorWithDefaultOptions allocator;
    for (int i = 0; i < numInputNodes; i++)
    {
        input_names.push_back(ort_session->GetInputName(i, allocator)); /// 低版本onnxruntime的接口函数
        ////AllocatedStringPtr input_name_Ptr = ort_session->GetInputNameAllocated(i, allocator);  /// 高版本onnxruntime的接口函数
        ////input_names.push_back(input_name_Ptr.get()); /// 高版本onnxruntime的接口函数
        Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_node_dims.push_back(input_dims);
    }
    for (int i = 0; i < numOutputNodes; i++)
    {
        output_names.push_back(ort_session->GetOutputName(i, allocator)); /// 低版本onnxruntime的接口函数
        ////AllocatedStringPtr output_name_Ptr= ort_session->GetInputNameAllocated(i, allocator);
        ////output_names.push_back(output_name_Ptr.get()); /// 高版本onnxruntime的接口函数
        Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_node_dims.push_back(output_dims);
    }

    this->input_height = input_node_dims[0][2];
    this->input_width = input_node_dims[0][3];
}

void Face68Landmarks::preprocess(Mat srcimg, const Bbox bounding_box)
{
    float sub_max = max(bounding_box.xmax - bounding_box.xmin, bounding_box.ymax - bounding_box.ymin);
    const float scale = 195.f / sub_max;
    const float translation[2] = {(256.f - (bounding_box.xmax + bounding_box.xmin) * scale) * 0.5f, (256.f - (bounding_box.ymax + bounding_box.ymin) * scale) * 0.5f};
    ////python程序里的warp_face_by_translation函数////
    Mat affine_matrix = (Mat_<float>(2, 3) << scale, 0.f, translation[0], 0.f, scale, translation[1]);
    Mat crop_img;
    warpAffine(srcimg, crop_img, affine_matrix, Size(256, 256));
    ////python程序里的warp_face_by_translation函数////
    cv::invertAffineTransform(affine_matrix, this->inv_affine_matrix);

    vector<cv::Mat> bgrChannels(3);
    split(crop_img, bgrChannels);
    for (int c = 0; c < 3; c++)
    {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / 255.0);
    }

    const int image_area = this->input_height * this->input_width;
    this->input_image.resize(3 * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    memcpy(this->input_image.data(), (float *)bgrChannels[0].data, single_chn_size);
    memcpy(this->input_image.data() + image_area, (float *)bgrChannels[1].data, single_chn_size);
    memcpy(this->input_image.data() + image_area * 2, (float *)bgrChannels[2].data, single_chn_size);
}

vector<Point2f> Face68Landmarks::detect(Mat srcimg, const Bbox bounding_box, vector<Point2f> &face_landmark_5of68)
{
    this->preprocess(srcimg, bounding_box);

    std::vector<int64_t> input_img_shape = {1, 3, this->input_height, this->input_width};
    Value input_tensor_ = Value::CreateTensor<float>(memory_info_handler, this->input_image.data(), this->input_image.size(), input_img_shape.data(), input_img_shape.size());

    Ort::RunOptions runOptions;
    vector<Value> ort_outputs = this->ort_session->Run(runOptions, this->input_names.data(), &input_tensor_, 1, this->output_names.data(), output_names.size());

    float *pdata = ort_outputs[0].GetTensorMutableData<float>(); /// 形状是(1, 68, 3), 每一行的长度是3，表示一个关键点坐标x,y和置信度
    const int num_points = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1];
    vector<Point2f> face_landmark_68(num_points);
    for (int i = 0; i < num_points; i++)
    {
        float x = pdata[i * 3] / 64.0 * 256.0;
        float y = pdata[i * 3 + 1] / 64.0 * 256.0;
        face_landmark_68[i] = Point2f(x, y);
    }
    vector<Point2f> face68landmarks;
    cv::transform(face_landmark_68, face68landmarks, this->inv_affine_matrix);

    ////python程序里的convert_face_landmark_68_to_5函数////
    face_landmark_5of68.resize(5);
    float x = 0, y = 0;
    for (int i = 36; i < 42; i++) /// left_eye
    {
        x += face68landmarks[i].x;
        y += face68landmarks[i].y;
    }
    x /= 6;
    y /= 6;
    face_landmark_5of68[0] = Point2f(x, y); /// left_eye

    x = 0, y = 0;
    for (int i = 42; i < 48; i++) /// right_eye
    {
        x += face68landmarks[i].x;
        y += face68landmarks[i].y;
    }
    x /= 6;
    y /= 6;
    face_landmark_5of68[1] = Point2f(x, y); /// right_eye

    face_landmark_5of68[2] = face68landmarks[30]; /// nose
    face_landmark_5of68[3] = face68landmarks[48]; /// left_mouth_end
    face_landmark_5of68[4] = face68landmarks[54]; /// right_mouth_end
    ////python程序里的convert_face_landmark_68_to_5函数////
    return face68landmarks;
}