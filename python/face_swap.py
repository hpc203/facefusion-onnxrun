import numpy as np
import onnxruntime
from utils import warp_face_by_face_landmark_5, create_static_box_mask, paste_back

FACE_MASK_BLUR = 0.3
FACE_MASK_PADDING = (0, 0, 0, 0)
INSWAPPER_128_MODEL_MEAN = [0.0, 0.0, 0.0]
INSWAPPER_128_MODEL_STD = [1.0, 1.0, 1.0]

class swap_face:
    def __init__(self, modelpath):
        # Initialize model
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3
        # self.session = onnxruntime.InferenceSession(modelpath, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.session = onnxruntime.InferenceSession(modelpath, sess_options=session_option)  ###opencv-dnn读取onnx失败
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = int(self.input_shape[2])
        self.input_width = int(self.input_shape[3])
        self.model_matrix = np.load('model_matrix.npy')

    def process(self, target_img, source_face_embedding, target_landmark_5):
        ###preprocess
        crop_img, affine_matrix = warp_face_by_face_landmark_5(target_img, target_landmark_5, 'arcface_128_v2', (128, 128))
        crop_mask_list = []

        box_mask = create_static_box_mask((crop_img.shape[1],crop_img.shape[0]), FACE_MASK_BLUR, FACE_MASK_PADDING)
        crop_mask_list.append(box_mask)

        crop_img = crop_img[:, :, ::-1].astype(np.float32) / 255.0
        crop_img = (crop_img - INSWAPPER_128_MODEL_MEAN) / INSWAPPER_128_MODEL_STD
        crop_img = np.expand_dims(crop_img.transpose(2, 0, 1), axis = 0).astype(np.float32)

        source_embedding = source_face_embedding.reshape((1, -1))
        source_embedding = np.dot(source_embedding, self.model_matrix) / np.linalg.norm(source_embedding)

        ###Perform inference on the image
        result = self.session.run(None, {'target':crop_img, 'source':source_embedding})[0][0]
        ###normalize_crop_frame
        result = result.transpose(1, 2, 0)
        result = (result * 255.0).round()
        result = result[:, :, ::-1]

        crop_mask = np.minimum.reduce(crop_mask_list).clip(0, 1)   ###print(np.array_equal(np.minimum.reduce(crop_mask_list), crop_mask_list[0])) 打印是True，说明np.minimum.reduce(crop_mask_list)等于crop_mask_list[0]，也就是box_mask，因此做np.minimum.reduce(crop_mask_list)完全是多此一举
        dstimg = paste_back(target_img, result, crop_mask, affine_matrix)
        return dstimg