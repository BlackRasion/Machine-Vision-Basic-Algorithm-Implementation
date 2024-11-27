import cv2 as cv
import numpy as np
def morphology_process(image, sub_option):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # 转换为灰度图像
    _, dog_thresh = cv.threshold(image_gray, 150, 255, cv.THRESH_BINARY) # 二值化
    

    if sub_option == '膨胀':
        kernel = np.ones((5, 5), np.uint8) # 5x5的卷积核
        dog_morphology = cv.dilate(dog_thresh, kernel, iterations=1)
    elif sub_option == '腐蚀':
        kernel = np.ones((5, 5), np.uint8) # 5x5的卷积核
        dog_morphology = cv.erode(dog_thresh, kernel, iterations=1) # 腐蚀
    elif sub_option == '开运算':
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        dog_morphology = cv.morphologyEx(dog_thresh, cv.MORPH_OPEN, kernel) # 开运算
    elif sub_option == '闭运算':
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        dog_morphology = cv.morphologyEx(dog_thresh, cv.MORPH_CLOSE, kernel)
    elif sub_option == '形态学梯度':
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        dog_morphology = cv.morphologyEx(dog_thresh, cv.MORPH_GRADIENT, kernel)
    elif sub_option == '顶帽':
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        dog_morphology = cv.morphologyEx(dog_thresh, cv.MORPH_TOPHAT, kernel)
    elif sub_option == '黑帽':
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        dog_morphology = cv.morphologyEx(dog_thresh, cv.MORPH_BLACKHAT, kernel)
    
    return dog_morphology
