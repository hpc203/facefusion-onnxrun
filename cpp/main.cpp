#include "yolov8face.h"
#include "face68landmarks.h"
#include "facerecognizer.h"
#include "faceswap.h"
#include "faceenhancer.h"

using namespace cv;
using namespace std;

int main()
{
	string source_path = "/project/faceswap-cpp/images/5.jpg";
	string target_path = "/project/faceswap-cpp/images/target.jpg";
    
	////图片路径和onnx文件的路径，要确保写正确，才能使程序正常运行的
	Yolov8Face detect_face_net("/project/faceswap-cpp/weights/yoloface_8n.onnx");
	Face68Landmarks detect_68landmarks_net("/project/faceswap-cpp/weights/2dfan4.onnx");
	FaceEmbdding face_embedding_net("/project/faceswap-cpp/weights/arcface_w600k_r50.onnx");
	SwapFace swap_face_net("/project/faceswap-cpp/weights/inswapper_128.onnx");
	FaceEnhance enhance_face_net("/project/faceswap-cpp/weights/gfpgan_1.4.onnx");

	Mat source_img = imread(source_path);
	Mat target_img = imread(target_path);

    vector<Bbox> boxes;
	detect_face_net.detect(source_img, boxes);
	int position = 0; ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况
	vector<Point2f> face_landmark_5of68;
	vector<Point2f> face68landmarks = detect_68landmarks_net.detect(source_img, boxes[position], face_landmark_5of68);
	vector<float> source_face_embedding = face_embedding_net.detect(source_img, face_landmark_5of68);

	detect_face_net.detect(target_img, boxes);
	position = 0; ////一张图片里可能有多个人脸，这里只考虑1个人脸的情况
	vector<Point2f> target_landmark_5;
	detect_68landmarks_net.detect(target_img, boxes[position], target_landmark_5);

	Mat swapimg = swap_face_net.process(target_img, source_face_embedding, target_landmark_5);
	Mat resultimg = enhance_face_net.process(swapimg, target_landmark_5);
	
	imwrite("resultimg.jpg", resultimg);

	/*static const string kWinName = "Deep learning face swap use onnxruntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, resultimg);
	waitKey(0);
	destroyAllWindows();*/
}