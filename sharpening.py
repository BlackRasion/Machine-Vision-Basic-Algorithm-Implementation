import cv2 as cv
import numpy as np
from utils import convolve2d

def sobel_sharpen(image):
    """使用Sobel算子进行锐化"""
    blurred = cv.GaussianBlur(image, (5, 5), 0) # 高斯滤波，去噪
    grad_x = cv.Sobel(blurred, cv.CV_64F, 1, 0, ksize=3) # 求x方向梯度
    grad_y = cv.Sobel(blurred, cv.CV_64F, 0, 1, ksize=3) # 求y方向梯度
    gradient_magnitude = cv.magnitude(grad_x, grad_y) # 梯度幅值
    gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude)) # 归一化
    sharpened_image = cv.addWeighted(image, 1.5, gradient_magnitude, 0.5, 0) # 叠加

    return sharpened_image

def laplacian_sharpen(lena):
    """使用拉普拉斯算子进行锐化"""
    lena_blurred_gaussian = cv.GaussianBlur(lena, (5, 5), 0) # 高斯滤波，去噪
    laplacian = cv.Laplacian(lena_blurred_gaussian, cv.CV_64F) # 拉普拉斯算子锐化
    laplacian = cv.convertScaleAbs(laplacian)
    lena_laplacian = cv.addWeighted(lena, 1.5, laplacian, -0.5, 0) # 叠加结合原图和锐化结果
    return lena_laplacian

def My_sobel_sharpen(image):
    """使用Sobel算子对图像进行锐化."""
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # 转为灰度图
    image_blurred = cv.GaussianBlur(image_gray, (3, 3), 0) # 高斯滤波，去噪
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

    # 增强边缘：将梯度幅值与原始图像相加
    sharpened_image = image_blurred.astype(float) + gradient_magnitude

    # 归一化，确保输出图像的数值范围保持在0-255之间
    sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)

    return sharpened_image