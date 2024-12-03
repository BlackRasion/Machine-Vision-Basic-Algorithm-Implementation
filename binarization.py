import cv2 as cv
import numpy as np
import ipywidgets as widgets

# 阈值滑动条，默认隐藏
threshold_slider = widgets.IntSlider(
    min=0,
    max=255,
    step=1,
    value=127,
    description='阈值:',
    layout={'visibility': 'hidden'}
)

def fixed_threshold(image, threshold=127):
    """使用固定阈值进行二值化"""
    _, binary_image = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
    return binary_image

def adaptive_mean_threshold(image, block_size=11, C=5):
    """使用自适应均值阈值进行二值化"""
    return cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block_size, C)

def adaptive_gaussian_threshold(image, block_size=11, C=5):
    """使用自适应高斯阈值进行二值化"""
    return cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block_size, C)

def otsu_threshold(image):
    """使用Otsu方法自动计算阈值进行二值化"""
    _, binary_image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return binary_image


# 自设otsu阈值分割
def My_otsu_threshold(image):
    # 计算图像的直方图
    hist, bin_edges = np.histogram(image, bins=range(257))
    
    # 归一化直方图
    hist_norm = hist / float(np.sum(hist))
    
    # 累积和 (cumulative sum)
    omega = np.cumsum(hist_norm)
    
    # 累积均值 (cumulative mean)
    mu = np.cumsum(hist_norm * np.arange(256))
    
    # 总均值水平
    mu_t = mu[-1]
    
    # 初始化
    sigma_b_squared = 0
    threshold = 0
    
    # 遍历所有可能的阈值
    for t in range(1, 256):
        # 类间方差
        omega_b = omega[t-1]
        omega_f = 1 - omega_b
        
        if omega_b == 0 or omega_f == 0:
            continue
        
        mu_b = mu[t-1] / omega_b
        mu_f = (mu_t - mu_b * omega_b) / omega_f
        
        # 计算类间方差
        _sigma_b_squared = omega_b * (1 - omega_b) * (mu_b - mu_f)**2
        
        # 更新最大类间方差和对应的阈值
        if _sigma_b_squared > sigma_b_squared:
            sigma_b_squared = _sigma_b_squared
            threshold = t
            
    print(f"Optimal threshold: {threshold}")

    # 应用阈值进行分割
    binary_image = np.zeros_like(image, dtype=np.uint8)
    binary_image[image > threshold] = 255
    return binary_image