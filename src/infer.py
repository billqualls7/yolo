'''
Author: wuyao sss
Date: 2024-02-22 17:11:14
LastEditors: wuyao sss
LastEditTime: 2024-02-23 15:00:29
FilePath: /rqh/YOLOv8/src/infer.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import math


def line_k(x1, y1, x2, y2):
    # 计算线段的斜率
    k = (y2 - y1) / (x2 - x1)

    
    return k


categories = ['pointer', 'digital']

model = YOLO("../dashboardbest.pt")
# img_path = '../img/76.jpg'
img_path = '../img/76.jpg'
img = cv2.imread(img_path)
# results = model([img_path], stream=True)  # return a generator of Results objects
results = model.predict(img_path, save=False, 
                        imgsz=640, conf=0.5, 
                        visualize=False
                        )
'''
data: tensor([[ 26.3736,  18.5083, 394.1399, 399.2077,   0.9480,   0.0000]], device='cuda:0')
分别代表xyxy conf cls 
'''
for result in results:
    boxes = result.boxes.data.cpu().numpy()  # Boxes object for bounding box outputs
    # data = boxes.data.cpu().numpy()

    for i in range(len(boxes)):
        data = boxes[i]
        x1 = data[0]
        y1 = data[1]
        x2 = data[2]
        y2 = data[3]
        score = data[4]
        cls = categories[int(data[5])]
        cropped_image = img[int(y1):int(y2), int(x1):int(x2)]
        H = abs(int(y2) - int(y1))
        W = abs(int(x2) - int(x1))

        gray_img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        # 整圆表盘用
        blurred_img = cv2.GaussianBlur(cropped_image, (3, 3), 0)
        gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
        threshold_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)
        # 1/4圆表盘用
        _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, hier = cv2.findContours(threshold_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)#轮廓查找

        edges = cv2.Canny(threshold_img, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 55)
        lines = lines[:, 0, :]
        result = edges.copy()
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + result.shape[1] * (-b))
            y1 = int(y0 + result.shape[1] * a)
            x2 = int(x0 - result.shape[0] * (-b))
            y2 = int(y0 - result.shape[0] * a)
            # 零刻度线
            if y1 >= H / 20 and y1 < H * 1 / 3:
                k1 = line_k(x1, y1, x2, y2)
                # cv2.line(threshold_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 指针
            if y2 >= H / 2 and y2 <= H * 4 / 5 and x2 >= H / 8:
                k2 = line_k(x1, y1, x2, y2)
                # cv2.line(threshold_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        Cobb = int(math.fabs(np.arctan((k1 - k2) / (float(1 + k1 * k2))) * 180 / np.pi) + 0.5)
        print("度数为：", int(Cobb * 1.6 / 270))




        # read_pointer(cropped_image)


        cv2.imwrite('img.jpg', (img))
        cv2.imwrite('cropped_image.jpg', (cropped_image))
        cv2.imwrite('gray_img.jpg', (gray_img))
        cv2.imwrite('binary_img.jpg', (binary_img))
        cv2.imwrite('threshold_img.jpg', (threshold_img))
        cv2.imwrite('edges.jpg', (edges))


    # print(boxes.to().cpu())
        # print((data))

