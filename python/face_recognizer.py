import cv2
import numpy as np
import onnxruntime
from utils import warp_face_by_face_landmark_5

class face_recognize:
    def __init__(self, modelpath):
        # Initialize model
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3
        self.session = onnxruntime.InferenceSession(modelpath, providers=['CPUExecutionProvider'])
        # self.session = onnxruntime.InferenceSession(modelpath, sess_options=session_option)  ###opencv-dnn读取onnx失败
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = int(self.input_shape[2])
        self.input_width = int(self.input_shape[3])

    def preprocess(self, srcimg, face_landmark_5):
        crop_img, _ = warp_face_by_face_landmark_5(srcimg, face_landmark_5, 'arcface_112_v2', (112, 112))
        crop_img = crop_img / 127.5 - 1
        crop_img = crop_img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
        crop_img = np.expand_dims(crop_img, axis = 0)
        return crop_img

    def detect(self, srcimg, face_landmark_5):
        input_tensor = self.preprocess(srcimg, face_landmark_5)

        # Perform inference on the image
        embedding = self.session.run(None, {self.input_names[0]: input_tensor})[0]
        embedding = embedding.ravel()  ###拉平
        normed_embedding = embedding / np.linalg.norm(embedding)
        return embedding, normed_embedding
    
if __name__ == '__main__':
    imgpath = '5.jpg'
    srcimg = cv2.imread('5.jpg')
    face_landmark_5 = np.array([[568.2485,  398.9512 ],
                            [701.7346,  399.64795],
                            [634.2213,  482.92694],
                            [583.5656,  543.10187],
                            [684.52405, 543.125  ]])
    
    mynet = face_recognize('weights/arcface_w600k_r50.onnx')
    embedding, normed_embedding = mynet.detect(srcimg, face_landmark_5)
    print(embedding.shape, normed_embedding.shape)