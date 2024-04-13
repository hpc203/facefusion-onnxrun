#include "utils.h"

using namespace std;
using namespace cv;

float GetIoU(const Bbox box1, const Bbox box2)
{
    float x1 = max(box1.xmin, box2.xmin);
    float y1 = max(box1.ymin, box2.ymin);
    float x2 = min(box1.xmax, box2.xmax);
    float y2 = min(box1.ymax, box2.ymax);
    float w = max(0.f, x2 - x1);
    float h = max(0.f, y2 - y1);
    float over_area = w * h;
    if (over_area == 0)
        return 0.0;
    float union_area = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin) + (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin) - over_area;
    return over_area / union_area;
}

vector<int> nms(vector<Bbox> boxes, vector<float> confidences, const float nms_thresh)
{
    sort(confidences.begin(), confidences.end(), [&confidences](size_t index_1, size_t index_2)
         { return confidences[index_1] > confidences[index_2]; });
    const int num_box = confidences.size();
    vector<bool> isSuppressed(num_box, false);
    for (int i = 0; i < num_box; ++i)
    {
        if (isSuppressed[i])
        {
            continue;
        }
        for (int j = i + 1; j < num_box; ++j)
        {
            if (isSuppressed[j])
            {
                continue;
            }

            float ovr = GetIoU(boxes[i], boxes[j]);
            if (ovr > nms_thresh)
            {
                isSuppressed[j] = true;
            }
        }
    }

    vector<int> keep_inds;
    for (int i = 0; i < isSuppressed.size(); i++)
    {
        if (!isSuppressed[i])
        {
            keep_inds.emplace_back(i);
        }
    }
    return keep_inds;
}

Mat warp_face_by_face_landmark_5(const Mat temp_vision_frame, Mat &crop_img, const vector<Point2f> face_landmark_5, const vector<Point2f> normed_template, const Size crop_size)
{
    vector<uchar> inliers(face_landmark_5.size(), 0);
    Mat affine_matrix = cv::estimateAffinePartial2D(face_landmark_5, normed_template, cv::noArray(), cv::RANSAC, 100.0);
    warpAffine(temp_vision_frame, crop_img, affine_matrix, crop_size, cv::INTER_AREA, cv::BORDER_REPLICATE);
    return affine_matrix;
}

Mat create_static_box_mask(const int *crop_size, const float face_mask_blur, const int *face_mask_padding)
{
    const float blur_amount = int(crop_size[0] * 0.5 * face_mask_blur);
    const int blur_area = max(int(blur_amount / 2), 1);
    Mat box_mask = Mat::ones(crop_size[0], crop_size[1], CV_32FC1);

    int sub = max(blur_area, int(crop_size[1] * face_mask_padding[0] / 100));
    // Mat roi = box_mask(cv::Rect(0,0,sub,crop_size[1]));
    box_mask(cv::Rect(0, 0, crop_size[1], sub)).setTo(0);

    sub = crop_size[0] - max(blur_area, int(crop_size[1] * face_mask_padding[2] / 100));
    box_mask(cv::Rect(0, sub, crop_size[1], crop_size[0] - sub)).setTo(0);

    sub = max(blur_area, int(crop_size[0] * face_mask_padding[3] / 100));
    box_mask(cv::Rect(0, 0, sub, crop_size[0])).setTo(0);

    sub = crop_size[1] - max(blur_area, int(crop_size[0] * face_mask_padding[1] / 100));
    box_mask(cv::Rect(sub, 0, crop_size[1] - sub, crop_size[0])).setTo(0);

    if (blur_amount > 0)
    {
        GaussianBlur(box_mask, box_mask, Size(0, 0), blur_amount * 0.25);
    }
    return box_mask;
}

Mat paste_back(Mat temp_vision_frame, Mat crop_vision_frame, Mat crop_mask, Mat affine_matrix)
{
    Mat inverse_matrix;
    cv::invertAffineTransform(affine_matrix, inverse_matrix);
    Mat inverse_mask;
    Size temp_size(temp_vision_frame.cols, temp_vision_frame.rows);
    warpAffine(crop_mask, inverse_mask, inverse_matrix, temp_size);
    inverse_mask.setTo(0, inverse_mask < 0);
    inverse_mask.setTo(1, inverse_mask > 1);
    Mat inverse_vision_frame;
    warpAffine(crop_vision_frame, inverse_vision_frame, inverse_matrix, temp_size, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    vector<Mat> inverse_vision_frame_bgrs(3);
    split(inverse_vision_frame, inverse_vision_frame_bgrs);
    vector<Mat> temp_vision_frame_bgrs(3);
    split(temp_vision_frame, temp_vision_frame_bgrs);
    for (int c = 0; c < 3; c++)
    {
        inverse_vision_frame_bgrs[c].convertTo(inverse_vision_frame_bgrs[c], CV_32FC1);   ////注意数据类型转换，不然在下面的矩阵点乘运算时会报错的
        temp_vision_frame_bgrs[c].convertTo(temp_vision_frame_bgrs[c], CV_32FC1);         ////注意数据类型转换，不然在下面的矩阵点乘运算时会报错的
    }
    vector<Mat> channel_mats(3);
    
    channel_mats[0] = inverse_mask.mul(inverse_vision_frame_bgrs[0]) + temp_vision_frame_bgrs[0].mul(1 - inverse_mask);
    channel_mats[1] = inverse_mask.mul(inverse_vision_frame_bgrs[1]) + temp_vision_frame_bgrs[1].mul(1 - inverse_mask);
    channel_mats[2] = inverse_mask.mul(inverse_vision_frame_bgrs[2]) + temp_vision_frame_bgrs[2].mul(1 - inverse_mask);
    
    cv::Mat paste_vision_frame;
    merge(channel_mats, paste_vision_frame);
    paste_vision_frame.convertTo(paste_vision_frame, CV_8UC3);
    return paste_vision_frame;
}

Mat blend_frame(Mat temp_vision_frame, Mat paste_vision_frame, const int FACE_ENHANCER_BLEND)
{
    const float face_enhancer_blend = 1 - ((float)FACE_ENHANCER_BLEND / 100.f);
    Mat dstimg;
    cv::addWeighted(temp_vision_frame, face_enhancer_blend, paste_vision_frame, 1 - face_enhancer_blend, 0, dstimg);
    return dstimg;
}