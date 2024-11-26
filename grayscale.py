import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def grayscale(image):
    """将图像转换为灰度图像"""
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def histogram_equalization(image):
    """执行灰度直方图均衡化"""
    return cv.equalizeHist(image)

def linear_transformation(image, alpha=1.5, beta=15):
    """执行线性变换"""
    return cv.convertScaleAbs(image, alpha=alpha, beta=beta)

def log_transformation(image, c=1):
    """执行对数变换"""
    log_transformed_image = c * np.log(1 + image).astype(np.uint8)
    return (log_transformed_image / np.max(log_transformed_image) * 255).astype(np.uint8)

def exponential_transformation(image, c=1, gamma=0.5):
    """执行指数变换"""
    exp_transformed_image = c * np.power(image, gamma).astype(np.uint8)
    return (exp_transformed_image / np.max(exp_transformed_image) * 255).astype(np.uint8)

def plot_histogram(image, bins=256):
    """绘制图像的灰度直方图"""
    plt.hist(image.ravel(), bins=bins, range=[0, 256], color='r', alpha=0.7)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.show()