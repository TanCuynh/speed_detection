import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
import time
from math import dist
from datetime import datetime

model = YOLO('yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('veh2.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
# print(class_list)

count = 0

# Tracker
tracker = Tracker()

# Vị trí vùng quan tâm theo trục Oy trong video
cy1 = 322
cy2 = 368

offset = 6

vh_down = {}
counter = []

vh_up = {}
counter1 = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    # xác định phương tiện bằng model YOLOv8
    results = model.predict(frame)
    #   print(results)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    #    print(px)
    list = []

    for index, row in px.iterrows():
        # print(row)

        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list.append([x1, y1, x2, y2])
    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 1)

        # going DOWN

        if (cy + offset) > cy1 > (cy - offset):
            vh_down[id] = time.time()
        if id in vh_down:

            if (cy + offset) > cy2 > (cy - offset):
                elapsed_time = time.time() - vh_down[id]
                if counter.count(id) == 0:
                    counter.append(id)
                    distance = 10  # meters
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh)) + 'km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (0, 255, 255), 2)
                    print("Vehicle speed (GOING DOWN) >>>>", a_speed_kh)

                    # Ghi tốc độ và thời gian vào tệp .txt
                    with open('vehicle_speed.txt', 'a') as speed_file:
                        speed_file.write(f'Vehicle ID: {id}, Time: {datetime.now()}, Speed: {a_speed_kh} km/h\n')

        # going UP

        if (cy + offset) > cy2 > (cy - offset):
            vh_up[id] = time.time()
        if id in vh_up:

            if (cy + offset) > cy1 > (cy - offset):
                elapsed1_time = time.time() - vh_up[id]

                if counter1.count(id) == 0:
                    counter1.append(id)
                    distance1 = 10  # meters
                    a_speed_ms1 = distance1 / elapsed1_time
                    a_speed_kh1 = a_speed_ms1 * 3.6
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh1)) + 'km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (0, 255, 255), 2)
                    print("Vehicle speed (GOING UP) >>>>", a_speed_kh1)

                    with open('vehicle_speed.txt', 'a') as speed_file:
                        speed_file.write(f'Vehicle ID: {id}, Time: {datetime.now()}, Speed: {a_speed_kh1} km/h\n')

    # Vẽ các đoạn thẳng tượng trưng cho vùng quan tâm (xác định tốc độ phương tiện)
    cv2.line(frame, (274, cy1), (814, cy1), (0, 0, 255), 1)

    cv2.line(frame, (177, cy2), (927, cy2), (0, 0, 255), 1)

    # Đếm và hiển thị số phương tiện đi theo chiều đến và chiều đi
    d = (len(counter))
    u = (len(counter1))
    cv2.putText(frame, 'Going down: ' + str(d), (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, 'Going up: ' + str(u), (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
