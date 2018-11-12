from darkflow.net.build import TFNet
import cv2
import numpy as np
from requirement import set_yolo_weights

set_yolo_weights()

options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}
tfnet = TFNet(options)

# start camera
cap = cv2.VideoCapture(0)

class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplane',
               'sheep', 'sofa', 'train', 'tvmonitor']

num_classes = len(class_names)
class_colors = []
for i in range(0, num_classes):
    hue = 255 * i/num_classes
    col = np.zeros((1, 1, 3)).astype("uint8")
    col[0][0][0] = hue
    col[0][0][1] = 128
    col[0][0][2] = 255
    cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
    col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
    class_colors.append(col)

def main():

    while(True):

        # 動画からフレームの取得
        ret, frame = cap.read()
        result = tfnet.return_predict(frame)

        for item in result:
            tlx = item['topleft']['x']
            tly = item['topleft']['y']
            brx = item['bottomright']['x']
            bry = item['bottomright']['y']
            label = item['label']
            conf = item['confidence']

            if conf > 0.6:

                for i in class_names:
                    if label == i:
                        class_num = class_names.index(i)
                        break

                # Create Flame
                cv2.rectangle(frame, (tlx, tly), (brx, bry), class_colors[class_num], 2)

                # Create Label
                text = label + " " + ('%.2f' % conf)
                cv2.rectangle(frame, (tlx, tly - 15), (tlx + 100, tly + 5), class_colors[class_num], -1)
                cv2.putText(frame, text, (tlx, tly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        # Show
        cv2.imshow("Show Flame Image", frame)

        # Break when esc pushed
        k = cv2.waitKey(10)
        if k == ord('q'): break;

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
