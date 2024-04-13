import cv2
import matplotlib.pyplot as plt  ###如无则pip安装
from yolov8face import YOLOface_8n
from face_68landmarks import face_68_landmarks
from face_recognizer import face_recognize
from face_swap import swap_face
from face_enhancer import enhance_face

if __name__ == '__main__':
    source_path = 'images/1.jpg'
    target_path = 'images/5.jpg'
    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)
    
    detect_face_net = YOLOface_8n("weights/yoloface_8n.onnx")
    detect_68landmarks_net = face_68_landmarks("weights/2dfan4.onnx")
    face_embedding_net = face_recognize('weights/arcface_w600k_r50.onnx')
    swap_face_net = swap_face('weights/inswapper_128.onnx')
    enhance_face_net = enhance_face('weights/gfpgan_1.4.onnx')

    boxes, _, _ = detect_face_net.detect(source_img)
    position = 0  ###一张图片里可能有多个人脸，这里只考虑1个人脸的情况
    bounding_box = boxes[position]
    _, face_landmark_5of68 = detect_68landmarks_net.detect(source_img, bounding_box)
    source_face_embedding, _ = face_embedding_net.detect(source_img, face_landmark_5of68)

    boxes, _, _ = detect_face_net.detect(target_img)
    position = 0  ###一张图片里可能有多个人脸，这里只考虑1个人脸的情况
    bounding_box = boxes[position]
    _, target_landmark_5 = detect_68landmarks_net.detect(target_img, bounding_box)

    swapimg = swap_face_net.process(target_img, source_face_embedding, target_landmark_5)
    resultimg = enhance_face_net.process(swapimg, target_landmark_5)
    
    plt.subplot(1, 2, 1)
    plt.imshow(source_img[:,:,::-1])  ###plt库显示图像是RGB顺序
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(target_img[:,:,::-1])
    plt.axis('off')
    # plt.show()
    plt.savefig('source_target.jpg', dpi=600, bbox_inches='tight') ###保存高清图

    cv2.imwrite('result.jpg', resultimg)
    
    # cv2.namedWindow('resultimg', 0)
    # cv2.imshow('resultimg', resultimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



    
