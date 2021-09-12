import cv2
import numpy as np

def lay_ki_tu(img):
    # resize bien so 400x450
    img_new = cv2.resize(img, (450, 400))
    otsu = cv2.threshold(img_new, 0, 255, cv2.THRESH_OTSU)[1]
    # lay tat ca cac doi tuong co trong bien so
    doituong = []
    contours, _ = cv2.findContours(
        otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        doituong.append([x, y, w, h])

    # lọc lấy các số bỏ đối tượng không mong muốn
    doituong_loc = []
    for x, y, w, h in doituong:
        tile_h = h/img_new.shape[0]
        tile_w = w/img_new.shape[1]
        if 0.25 < tile_h < 0.4 and 0.04<tile_w<0.15:
            doituong_loc.append([x, y, w, h])
    # tính giá trị trung bình trục y và chia biển số thành phần trên và phần dưới
    mean_y = np.mean(np.array(doituong_loc)[:, 1])
    top = []
    bot = []
    for i in doituong_loc:
        if i[1] < mean_y:
            top.append(i)
        else:
            bot.append(i)
    # Sắp xếp các số theo thứ tự theo trục x
    top = sorted(top, key=lambda x: x[0])
    bot = sorted(bot, key=lambda x: x[0])

    # resize về 32x32 để predict
    img_bot = []
    img_top = []
    for x, y, w, h in top:
        im = otsu[y:y+h, x:x+w]
        im = 255-im
        # resize 28x28
        im = cv2.resize(im, (28, 28))
        # them border 4 huong 28 +4 = 32 -> 32x32
        im = cv2.copyMakeBorder(im, 2, 2, 2, 2, cv2.BORDER_CONSTANT)
        im = 255-im
        img_top.append(im)

    for x, y, w, h in bot:
        im = otsu[y:y+h, x:x+w]
        im = 255-im
        # resize 28x28
        im = cv2.resize(im, (28, 28))
        # them border 4 huong 28 +4 = 32 -> 32x32
        im = cv2.copyMakeBorder(im, 2, 2, 2, 2, cv2.BORDER_CONSTANT)
        im = 255-im
        img_bot.append(im)

    return img_top, img_bot
def predict(top,bot, model):
    labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H',
              18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'P', 25: 'Q', 26: 'R', 27: 'S', 28: 'T', 29: 'U', 30: 'V', 31: 'W', 32: 'X', 33: 'Y', 34: 'Z'}
    rs = ""
    for i in top:
        img = i.reshape(1, 32, 32, 1)
        pre = model.predict(img)
        result = labels[np.argmax(pre)]
        rs += result
    rs += "-"
    for i in bot:
        img = i.reshape(1, 32, 32, 1)
        pre = model.predict(img)
        result = labels[np.argmax(pre)]
        rs += result
    return rs


# phát biện và cắt biển số

# def cat_bien_so(img, net):
#     font = cv2.FONT_HERSHEY_PLAIN
#     height, width, _ = img.shape
#     blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
#     net.setInput(blob)
#     output_layers_names = net.getUnconnectedOutLayersNames()
#     layerOutputs = net.forward(output_layers_names)

#     boxes = []
#     confidences = []

#     for output in layerOutputs:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.2:
#                 center_x = int(detection[0]*width)
#                 center_y = int(detection[1]*height)
#                 w = int(detection[2]*width)
#                 h = int(detection[3]*height)

#                 x = int(center_x - w/2)
#                 y = int(center_y - h/2)

#                 boxes.append([x, y, w, h])
#                 confidences.append((float(confidence)))

#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

#     if len(indexes)>0:
#         for i in indexes.flatten():
#             x, y, w, h = boxes[i]
#             confidence = str(round(confidences[i],4))
#             bien_so = img[round(y):round(y+h), round(x):round(x+w)]
#             bien_so = cv2.cvtColor(bien_so, cv2.COLOR_BGR2GRAY)
#             top,bot = lay_ki_tu(bien_so)
#             rs = predict(top,bot,model)
            
#             cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
#             cv2.putText(img, confidence, (x, y+20), font, 2, (0,0,255), 2)
            

#     return img