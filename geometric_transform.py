import cv2 as cv
import numpy as np
from ipywidgets import widgets
from scipy.ndimage import affine_transform

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

def get_affine_matrix(rotation=0, scale=(1, 1), shear=10, translation=(0, 0)):
    """创建一个仿射变换矩阵."""
    # 旋转 (角度转换为弧度)
    theta = np.radians(rotation)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    # 缩放
    scale_matrix = np.array([[scale[0], 0, 0],
                             [0, scale[1], 0],
                             [0, 0, 1]])

    # 剪切
    shear_matrix = np.array([[1, np.tan(np.radians(shear)), 0],
                             [0, 1, 0],
                             [0, 0, 1]])

    # 组合所有变换（注意顺序：旋转 -> 缩放 -> 剪切）
    combined_matrix = rotation_matrix @ scale_matrix @ shear_matrix

    # 返回线性变换矩阵和平移向量
    return combined_matrix[:2, :2], translation

def apply_affine_transform(image, linear_matrix, translation_vector):
    """应用仿射变换到图像."""
    # 获取图像尺寸
    h, w = image.shape[:2]

    # 计算新的中心点偏移量
    center = np.array([w / 2, h / 2])
    # 先计算线性变换对中心点的影响
    transformed_center = linear_matrix.dot(center)
    # 然后计算总的偏移量
    offset = center - transformed_center + translation_vector

    # 应用仿射变换
    transformed_image = affine_transform(image, linear_matrix, offset=offset, output_shape=image.shape)

    return transformed_image

def My_affine_transform(image, rotation=0, scale=(1, 1), shear=10, translation=(0, 0)):
    """自定义仿射变换"""
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 获取仿射变换矩阵
    linear_matrix, translation_vector = get_affine_matrix(rotation, scale, shear, translation)

    # 应用仿射变换
    transformed_image = apply_affine_transform(image_gray, linear_matrix, translation_vector)

    return transformed_image