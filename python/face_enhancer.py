import numpy as np
import onnxruntime
from utils import warp_face_by_face_landmark_5, create_static_box_mask, paste_back, blend_frame

FACE_MASK_BLUR = 0.3
FACE_MASK_PADDING = (0, 0, 0, 0)

class enhance_face:
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

    def process(self, target_img, target_landmark_5):
        ###preprocess
        crop_img, affine_matrix = warp_face_by_face_landmark_5(target_img, target_landmark_5, 'ffhq_512', (512, 512))
        box_mask = create_static_box_mask((crop_img.shape[1],crop_img.shape[0]), FACE_MASK_BLUR, FACE_MASK_PADDING)
        crop_mask_list = [box_mask]
    
        crop_img = crop_img[:, :, ::-1].astype(np.float32) / 255.0
        crop_img = (crop_img - 0.5) / 0.5
        crop_img = np.expand_dims(crop_img.transpose(2, 0, 1), axis = 0).astype(np.float32)

        ###Perform inference on the image
        result = self.session.run(None, {'input':crop_img})[0][0]
        ###normalize_crop_frame
        result = np.clip(result, -1, 1)
        result = (result + 1) / 2
        result = result.transpose(1, 2, 0)
        result = (result * 255.0).round()
        result = result.astype(np.uint8)[:, :, ::-1]

        crop_mask = np.minimum.reduce(crop_mask_list).clip(0, 1)
        paste_frame = paste_back(target_img, result, crop_mask, affine_matrix)
        dstimg = blend_frame(target_img, paste_frame)
        return dstimg