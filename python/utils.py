import numpy as np 
import cv2

def warp_face_by_translation(temp_img, translation, scale, crop_size):
    affine_matrix = np.array([[ scale, 0, translation[0] ], [ 0, scale, translation[1] ]])
    crop_img = cv2.warpAffine(temp_img, affine_matrix, crop_size)
    return crop_img, affine_matrix

def convert_face_landmark_68_to_5(landmark_68):
    left_eye = np.mean(landmark_68[36:42], axis = 0)
    right_eye = np.mean(landmark_68[42:48], axis = 0)
    nose = landmark_68[30]
    left_mouth_end = landmark_68[48]
    right_mouth_end = landmark_68[54]
    face_landmark_5 = np.array([left_eye, right_eye, nose, left_mouth_end, right_mouth_end])
    return face_landmark_5

TEMPLATES = {'arcface_112_v2': np.array([[ 0.34191607, 0.46157411 ],
                                         [ 0.65653393, 0.45983393 ],
                                         [ 0.50022500, 0.64050536 ],
                                         [ 0.37097589, 0.82469196 ],
                                         [ 0.63151696, 0.82325089 ]]),
             'arcface_128_v2': np.array([[ 0.36167656, 0.40387734 ],
                                         [ 0.63696719, 0.40235469 ],
                                         [ 0.50019687, 0.56044219 ],
                                         [ 0.38710391, 0.72160547 ],
                                         [ 0.61507734, 0.72034453 ]]),
             'ffhq_512': np.array([[ 0.37691676, 0.46864664 ],
                                   [ 0.62285697, 0.46912813 ],
                                   [ 0.50123859, 0.61331904 ],
                                   [ 0.39308822, 0.72541100 ],
                                   [ 0.61150205, 0.72490465 ]])}

def warp_face_by_face_landmark_5(temp_vision_frame, face_landmark_5, template, crop_size):
    normed_template = TEMPLATES.get(template) * crop_size
    # print(normed_template)  ###打印出来，写到c++程序的std::vector<cv::Point2f> normed_template里
    affine_matrix = cv2.estimateAffinePartial2D(face_landmark_5, normed_template, method = cv2.RANSAC, ransacReprojThreshold = 100)[0]
    crop_img = cv2.warpAffine(temp_vision_frame, affine_matrix, crop_size, borderMode = cv2.BORDER_REPLICATE, flags = cv2.INTER_AREA)
    return crop_img, affine_matrix

def create_static_box_mask(crop_size, face_mask_blur, face_mask_padding):
    blur_amount = int(crop_size[0] * 0.5 * face_mask_blur)
    blur_area = max(blur_amount // 2, 1)
    box_mask = np.ones(crop_size, np.float32)
    box_mask[:max(blur_area, int(crop_size[1] * face_mask_padding[0] / 100)), :] = 0
    box_mask[-max(blur_area, int(crop_size[1] * face_mask_padding[2] / 100)):, :] = 0
    box_mask[:, :max(blur_area, int(crop_size[0] * face_mask_padding[3] / 100))] = 0
    box_mask[:, -max(blur_area, int(crop_size[0] * face_mask_padding[1] / 100)):] = 0
    if blur_amount > 0:
        box_mask = cv2.GaussianBlur(box_mask, (0, 0), blur_amount * 0.25)
    return box_mask

def paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix):
    inverse_matrix = cv2.invertAffineTransform(affine_matrix)
    temp_size = temp_vision_frame.shape[:2][::-1]
    inverse_mask = cv2.warpAffine(crop_mask, inverse_matrix, temp_size).clip(0, 1)
    inverse_vision_frame = cv2.warpAffine(crop_vision_frame, inverse_matrix, temp_size, borderMode = cv2.BORDER_REPLICATE)
    paste_vision_frame = temp_vision_frame.copy()
    paste_vision_frame[:, :, 0] = inverse_mask * inverse_vision_frame[:, :, 0] + (1 - inverse_mask) * temp_vision_frame[:, :, 0]
    paste_vision_frame[:, :, 1] = inverse_mask * inverse_vision_frame[:, :, 1] + (1 - inverse_mask) * temp_vision_frame[:, :, 1]
    paste_vision_frame[:, :, 2] = inverse_mask * inverse_vision_frame[:, :, 2] + (1 - inverse_mask) * temp_vision_frame[:, :, 2]
    return paste_vision_frame

def blend_frame(temp_vision_frame, paste_vision_frame, FACE_ENHANCER_BLEND=80):
    face_enhancer_blend = 1 - (FACE_ENHANCER_BLEND / 100)
    temp_vision_frame = cv2.addWeighted(temp_vision_frame, face_enhancer_blend, paste_vision_frame, 1 - face_enhancer_blend, 0)
    return temp_vision_frame