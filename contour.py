import cv2 as cv

def contours(img, sub_option, threshold):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # 转换为灰度图像
    img_blurred = cv.GaussianBlur(img_gray, (5, 5), 0) # 高斯滤波，去噪
    if sub_option == '轮廓绘制':
        # 二值化
        _, th_fixed = cv.threshold(img_blurred, threshold, 255, cv.THRESH_BINARY)

        contours, hierarchy = cv.findContours(th_fixed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # 查找轮廓
        # 绘制轮廓
        img_contours = img.copy()
        cv.drawContours(img_contours, contours, -1, (0, 255, 0), 1)

    return img_contours
