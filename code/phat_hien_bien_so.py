from extra import lay_ki_tu, predict
import cv2
import numpy as np
import time
import tensorflow as tf

with tf.device('/cpu:0'):
   model = tf.keras.models.load_model("../model/cnn_bienso.h5")
print("bat dau.................")
net = cv2.dnn.readNet('../model/yolov4-tiny.cfg', '../model/yolov4-tiny_3000.weights')

classes = []
# with open("coco.names", "r") as f:
#     classes = f.read().splitlines()

# this below two line will help to run the detetection.
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)

net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('../video/demo5.mp4')
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))
prev_frame_time = 0
new_frame_time = 0
count = 0
rs = ''
while True:
    _, img = cap.read()
    count+=1
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            bien_so = img[round(y):round(y+h), round(x):round(x+w)]
            try:
              bien_so = cv2.cvtColor(bien_so, cv2.COLOR_BGR2GRAY)
              top,bot = lay_ki_tu(bien_so)
      
              if count%20==0:
                pre = predict(top,bot,model)
                rs=pre
            except:
              pass
            confidence = str(round(confidences[i],4))
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(img, confidence+"    "+rs, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            
            # print(ima)
     # tinh fps
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    cv2.putText(img, "fps "+str(fps) , (20, 20 ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()