from ipywidgets import widgets
import cv2 as cv
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