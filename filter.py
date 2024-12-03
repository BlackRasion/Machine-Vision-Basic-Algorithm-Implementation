from ipywidgets import widgets
import cv2 as cv
import numpy as np
from scipy.ndimage import convolve

# 滤波器参数控件，默认隐藏
filter_params = widgets.HBox([
    widgets.IntSlider(min=1, max=15, step=2, value=5, description='Kernel Size:', layout={'visibility': 'hidden'}),
    widgets.FloatSlider(min=0.1, max=5.0, step=0.1, value=1.5, description='Sigma X:', layout={'visibility': 'hidden'}),
    widgets.IntSlider(min=1, max=15, step=2, value=9, description='Diameter:', layout={'visibility': 'hidden'}),
    widgets.IntSlider(min=1, max=100, step=1, value=75, description='Sigma Color:', layout={'visibility': 'hidden'}),
    widgets.IntSlider(min=1, max=100, step=1, value=75, description='Sigma Space:', layout={'visibility': 'hidden'})
])

# 均值滤波
def apply_mean_filter(image, kernel_size):
    kernel_size = kernel_size or 5
    return cv.blur(image, (kernel_size, kernel_size))

# 高斯滤波
def apply_gaussian_filter(image, kernel_size, sigma_x):
    kernel_size = kernel_size or 5
    sigma_x = sigma_x or 1.5
    return cv.GaussianBlur(image, (kernel_size, kernel_size), sigma_x)

# 自设高斯滤波
def gaussian_kernel(size, sigma=1):
    """生成一个高斯滤波器"""
    # 创建一个中心为0的坐标网格
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    
    # 计算每个位置的高斯值
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    
    # 归一化，使得核的总和为1
    kernel /= 2 * np.pi * sigma**2
    kernel /= np.sum(kernel)
    
    return kernel

def My_gaussian_filter(image, kernel_size, sigma):
    # 生成高斯核
    kernel = gaussian_kernel(kernel_size, sigma)
    # 使用scipy的convolve函数进行卷积
    filtered_image = convolve(image, kernel, mode='reflect')
    
    # 确保输出图像的数值范围保持在0-255之间
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    
    return filtered_image

# 中值滤波
def apply_median_filter(image, kernel_size):
    kernel_size = kernel_size or 5
    return cv.medianBlur(image, kernel_size)

# 双边滤波
def apply_bilateral_filter(image, diameter, sigma_color, sigma_space):
    diameter = diameter or 9
    sigma_color = sigma_color or 75
    sigma_space = sigma_space or 75
    return cv.bilateralFilter(image, diameter, sigma_color, sigma_space)