'''
Author: wuyao sss
Date: 2024-02-22 17:11:14
LastEditors: wuyao sss
LastEditTime: 2024-02-22 17:58:23
FilePath: /rqh/YOLOv8/src/infer.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("../dashboardbest.pt")

results = model(['../img/76.jpg'], stream=True)  # return a generator of Results objects

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    boxes.cls
    probs = result.probs  # Probs object for classification outputs
    print(boxes.cls)
    # print(result.names)
    # result.show()  # display to screen
    # result.save(filename='result.jpg')  # save to disk