import cv2 as cv
import ipywidgets as widgets
import numpy as np
from utils import convolve2d

# 边缘检测参数控件，默认隐藏
edge_detection_params = widgets.HBox([
    widgets.IntSlider(min=0, max=255, step=1, value=100, description='Low Threshold:', layout={'visibility': 'hidden'}),
    widgets.IntSlider(min=0, max=255, step=1, value=200, description='High Threshold:', layout={'visibility': 'hidden'}),
])

# Canny边缘检测
def canny_edge_detect(image, low_threshold, high_threshold):
    """使用Canny算子进行边缘检测"""
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.GaussianBlur(image, (5, 5), 0) # 高斯滤波，去噪
    return cv.Canny(image, low_threshold, high_threshold)

# Sobel边缘检测
def sobel_edge_detect(image):
    """使用Sobel算子进行边缘检测"""
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # 灰度化
    grad_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3) # 求x方向梯度
    grad_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3) # 求y方向梯度
    gradient_magnitude = cv.magnitude(grad_x, grad_y) # 梯度幅值
    gradient_magnitude = cv.convertScaleAbs(gradient_magnitude)
    return gradient_magnitude

# Laplacian边缘检测
def laplacian_edge_detect(image):
    """使用Laplacian算子进行边缘检测"""
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return cv.Laplacian(image, cv.CV_64F)

def My_sobel_edge_detect(image):
    # 转为灰度图并应用高斯滤波
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_blurred = cv.GaussianBlur(image_gray, (3, 3), 0)
    
    # 定义Sobel算子的卷积核
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=float)
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=float)
    
    # 计算水平和垂直方向的梯度
    grad_x = convolve2d(image_blurred, sobel_x)
    grad_y = convolve2d(image_blurred, sobel_y)
    
    # 计算梯度幅值
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # 归一化，确保输出图像的数值范围保持在0-255之间
    gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)
    
    return gradient_magnitude