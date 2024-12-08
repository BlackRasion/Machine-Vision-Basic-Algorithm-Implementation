import cv2 as cv
import ipywidgets as widgets

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
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
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