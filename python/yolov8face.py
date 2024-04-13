import cv2
import numpy as np
import onnxruntime
import argparse


class YOLOface_8n:
    def __init__(self, modelpath, conf_thres=0.5, iou_thresh=0.4):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thresh
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

    def preprocess(self, srcimg):
        height, width = srcimg.shape[:2]
        temp_image = srcimg.copy()
        if height > self.input_height or width > self.input_width:
            scale = min(self.input_height / height, self.input_width / width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            temp_image = cv2.resize(srcimg, (new_width, new_height))
        self.ratio_height = height / temp_image.shape[0]
        self.ratio_width = width / temp_image.shape[1]
        input_img = cv2.copyMakeBorder(temp_image, 0, self.input_height - temp_image.shape[0], 0, self.input_width - temp_image.shape[1], cv2.BORDER_CONSTANT,
                                         value=0)
        # Scale input pixel values to 0 to 1
        input_img = (input_img.astype(np.float32) - 127.5) / 128.0
        input_img = input_img.transpose(2, 0, 1)
        input_img = input_img[np.newaxis, :, :, :]
        return input_img

    def detect(self, srcimg):
        input_tensor = self.preprocess(srcimg)

        # Perform inference on the image
        outputs = self.session.run(None, {self.input_names[0]: input_tensor})[0]
        boxes, kpts, scores = self.postprocess(outputs)
        return boxes, kpts, scores

    def postprocess(self, outputs):
        bounding_box_list, face_landmark5_list, score_list= [], [], []
        
        outputs = np.squeeze(outputs, axis=0).T
        bounding_box_raw, score_raw, face_landmark_5_raw = np.split(outputs, [ 4, 5 ], axis = 1)
        keep_indices = np.where(score_raw > self.conf_threshold)[0]
        if keep_indices.any():
            bounding_box_raw, face_landmark_5_raw, score_raw = bounding_box_raw[keep_indices], face_landmark_5_raw[keep_indices], score_raw[keep_indices]
            bboxes_wh = bounding_box_raw.copy()
            bboxes_wh[:, :2] = bounding_box_raw[:, :2] - 0.5 * bounding_box_raw[:, 2:]  ####(cx,cy,w,h)转到(x,y,w,h)
            bboxes_wh *= np.array([[self.ratio_width, self.ratio_height, self.ratio_width, self.ratio_height]])  ###合理使用广播法则
            face_landmark_5_raw *= np.tile(np.array([self.ratio_width, self.ratio_height, 1]), 5).reshape((1, 15))  ###合理使用广播法则,每个点的信息是(x,y,conf), 第3个元素点的置信度，可以不要，那也就需要要乘以1
            score_raw = score_raw.flatten()

            indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), score_raw.tolist(), self.conf_threshold, self.iou_threshold)
            if isinstance(indices, np.ndarray):
                indices = indices.flatten()
            if len(indices) > 0:
                # bounding_box_list = list(bboxes_wh[indices])
                bounding_box_list = list(map(lambda x:np.array([x[0], x[1], x[0]+x[2], x[1]+x[3]], dtype=np.float64), bboxes_wh[indices])) ###xywh转到xminyminxmaxymax
                score_list = list(score_raw[indices])
                face_landmark5_list = list(face_landmark_5_raw[indices])

        return bounding_box_list, face_landmark5_list, score_list

    def draw_detections(self, image, boxes,  kpts, scores):
        for box, kp, score in zip(boxes, kpts, scores):
            xmin, ymin, xmax, ymax = box.astype(int)
            
            # Draw rectangle
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
            label = "face:"+str(round(score,2))
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
            for i in range(5):
                cv2.circle(image, (int(kp[i * 3]), int(kp[i * 3 + 1])), 3, (0, 255, 0), thickness=-1)
        return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='5.jpg', help="image path")
    parser.add_argument('--confThreshold', default=0.5, type=float, help='class confidence')
    args = parser.parse_args()

    # Initialize YOLOface_8n object detector
    mynet = YOLOface_8n("weights/yoloface_8n.onnx", conf_thres=args.confThreshold)
    srcimg = cv2.imread(args.imgpath)

    # Detect Objects
    boxes, kpts, scores = mynet.detect(srcimg)

    # Draw detections
    dstimg = mynet.draw_detections(srcimg, boxes, kpts, scores)
    winName = 'Deep learning yolov8face detection in ONNXRuntime'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, dstimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
