import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def grayscale(image):
    """将图像转换为灰度图像"""
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def histogram_equalization(image):
    """执行灰度直方图均衡化"""
    return cv.equalizeHist(image)

def My_hist_equalization(image):
    # 确保输入是灰度图像
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 计算图像的直方图
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # 计算累积分布函数
    cdf = hist.cumsum()
    # 归一化累积分布函数
    cdf_normalized = cdf * hist.max() / cdf.max()

    # 忽略直方图中的零值
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    # 填充被忽略的值为0
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    # 将原始图像映射到新的灰度值
    img_eq = cdf[image]

    return img_eq

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