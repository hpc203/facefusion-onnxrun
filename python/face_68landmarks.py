import cv2
import numpy as np
import onnxruntime
from utils import warp_face_by_translation, convert_face_landmark_68_to_5

class face_68_landmarks:
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


    def preprocess(self, srcimg, bounding_box):
        '''
        bounding_box里的数据格式是[xmin. ymin, xmax, ymax]
        '''
        scale = 195 / np.subtract(bounding_box[2:], bounding_box[:2]).max()
        translation = (256 - np.add(bounding_box[2:], bounding_box[:2]) * scale) * 0.5
        crop_img, affine_matrix = warp_face_by_translation(srcimg, translation, scale, (256, 256))

        # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2Lab)  ###可有可无
        # if np.mean(crop_img[:, :, 0]) < 30:
        #     crop_img[:, :, 0] = cv2.createCLAHE(clipLimit = 2).apply(crop_img[:, :, 0])
        # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_Lab2RGB)   ###可有可无
        
        crop_img = crop_img.transpose(2, 0, 1).astype(np.float32) / 255.0
        crop_img = crop_img[np.newaxis, :, :, :]
        return crop_img, affine_matrix

    def detect(self, srcimg, bounding_box):
        '''
        如果直接crop+resize,最后返回的人脸关键点有偏差
        '''
        input_tensor, affine_matrix = self.preprocess(srcimg, bounding_box)

        # Perform inference on the image
        face_landmark_68 = self.session.run(None, {self.input_names[0]: input_tensor})[0]
        face_landmark_68 = face_landmark_68[:, :, :2][0] / 64
        face_landmark_68 = face_landmark_68.reshape(1, -1, 2) * 256
        face_landmark_68 = cv2.transform(face_landmark_68, cv2.invertAffineTransform(affine_matrix))
        face_landmark_68 = face_landmark_68.reshape(-1, 2)
        face_landmark_5of68 = convert_face_landmark_68_to_5(face_landmark_68)
        return face_landmark_68, face_landmark_5of68

if __name__ == '__main__':
    imgpath = '5.jpg'
    srcimg = cv2.imread('5.jpg')
    bounding_box = np.array([487, 236, 784, 624])
    
    # Initialize face_68landmarks detector
    mynet = face_68_landmarks("weights/2dfan4.onnx")

    face_landmark_68, face_landmark_5of68 = mynet.detect(srcimg, bounding_box)
    # print(face_landmark_5of68)
    # Draw detections
    for i in range(face_landmark_68.shape[0]):
        cv2.circle(srcimg, (int(face_landmark_68[i,0]), int(face_landmark_68[i,1])), 3, (0, 255, 0), thickness=-1)
    cv2.imwrite('detect_face_68lanmarks.jpg', srcimg)
    winName = 'Deep learning face_68landmarks detection in ONNXRuntime'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
