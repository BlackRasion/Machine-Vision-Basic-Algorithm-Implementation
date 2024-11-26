import cv2 as cv
import numpy as np
from ipywidgets import widgets

# 几何变换参数控件，默认隐藏
geometry_params = widgets.HBox([
    widgets.FloatSlider(min=-100, max=100, step=1, value=0, description='Translation X:', layout={'visibility': 'hidden'}),
    widgets.FloatSlider(min=-100, max=100, step=1, value=0, description='Translation Y:', layout={'visibility': 'hidden'}),
    widgets.FloatSlider(min=-180, max=180, step=1, value=0, description='Rotation Angle:', layout={'visibility': 'hidden'}),
    widgets.FloatSlider(min=0.1, max=3, step=0.1, value=1, description='Scale Factor:', layout={'visibility': 'hidden'}),
    widgets.Dropdown(options=['None', 'Horizontal', 'Vertical', 'Both'], value='None', description='Flip Direction:', layout={'visibility': 'hidden'})
])

def translate(image, tx, ty):
    """平移图像"""
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv.warpAffine(image, M, (cols, rows))
    return translated_image

def rotate(image, angle):
    """旋转图像"""
    rows, cols = image.shape[:2]
    M = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated_image = cv.warpAffine(image, M, (cols, rows))
    return rotated_image

def shear(image, sx=1, sy=0):
    """错切图像"""
    rows, cols = image.shape[:2]
    M = np.float32([[1, sx, 0], [sy, 1, 0]])
    sheared_image = cv.warpAffine(image, M, (cols, rows))
    return sheared_image

def scale(image, scale_factor):
    """缩放图像"""
    fx = scale_factor
    fy = scale_factor
    scaled_image = cv.resize(image, None, fx=fx, fy=fy, interpolation=cv.INTER_LINEAR)
    return scaled_image

def flip(image, direction):
    """翻转图像"""
    if direction == 'Horizontal':
        flipped_image = cv.flip(image, 1)
    elif direction == 'Vertical':
        flipped_image = cv.flip(image, 0)
    elif direction == 'Both':
        flipped_image = cv.flip(image, -1)
    else:
        flipped_image = image  # 如果方向为None，返回原图像
    return flipped_image