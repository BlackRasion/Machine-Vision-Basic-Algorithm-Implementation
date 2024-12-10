import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 低通滤波
def apply_low_pass_filter(image, cutoff_frequency):
    # 获取图像的高度和宽度
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # 创建一个与图像大小相同的掩码
    mask = np.zeros((rows, cols, 2), np.uint8)
    
    # 在中心区域创建一个圆形掩码
    r = int(cutoff_frequency)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - crow) ** 2 + (y - ccol) ** 2 <= r * r
    mask[mask_area] = 1
    
    # 进行傅里叶变换
    f_transform = cv.dft(image, flags=cv.DFT_COMPLEX_OUTPUT)
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    # 应用低通滤波器
    filtered_f_transform = f_transform_shifted * mask
    
    # 进行逆傅里叶变换
    filtered_f_transform_ishift = np.fft.ifftshift(filtered_f_transform)
    filtered_image = cv.idft(filtered_f_transform_ishift)
    filtered_image = cv.magnitude(filtered_image[:, :, 0], filtered_image[:, :, 1])
    
    return filtered_image

# 高通滤波
def apply_high_pass_filter(image, cutoff_frequency):
    # 获取图像的高度和宽度
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # 创建一个与图像大小相同的掩码
    mask = np.ones((rows, cols, 2), np.uint8)
    
    # 在中心区域创建一个圆形掩码
    r = int(cutoff_frequency)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - crow) ** 2 + (y - ccol) ** 2 <= r * r
    mask[mask_area] = 0
    
    # 进行傅里叶变换
    f_transform = cv.dft(image, flags=cv.DFT_COMPLEX_OUTPUT)
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    # 应用高通滤波器
    filtered_f_transform = f_transform_shifted * mask
    
    # 进行逆傅里叶变换
    filtered_f_transform_ishift = np.fft.ifftshift(filtered_f_transform)
    filtered_image = cv.idft(filtered_f_transform_ishift)
    filtered_image = cv.magnitude(filtered_image[:, :, 0], filtered_image[:, :, 1])
    
    return filtered_image

# 带阻(陷波)滤波
def apply_band_stop_filter(image, low_cutoff, high_cutoff):
    # 获取图像的高度和宽度
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # 创建一个与图像大小相同的掩码
    mask = np.ones((rows, cols, 2), np.uint8)
    
    # 在中心区域创建两个圆形掩码
    low_r = int(low_cutoff)
    high_r = int(high_cutoff)
    x, y = np.ogrid[:rows, :cols]
    low_mask_area = (x - crow) ** 2 + (y - ccol) ** 2 <= low_r * low_r
    high_mask_area = (x - crow) ** 2 + (y - ccol) ** 2 >= high_r * high_r
    mask[low_mask_area | high_mask_area] = 0
    
    # 进行傅里叶变换
    f_transform = cv.dft(image, flags=cv.DFT_COMPLEX_OUTPUT)
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    # 应用带阻滤波器
    filtered_f_transform = f_transform_shifted * mask
    
    # 进行逆傅里叶变换
    filtered_f_transform_ishift = np.fft.ifftshift(filtered_f_transform)
    filtered_image = cv.idft(filtered_f_transform_ishift)
    filtered_image = cv.magnitude(filtered_image[:, :, 0], filtered_image[:, :, 1])
    
    return filtered_image

# 频率滤波
def frequency_filter(image, sub_option):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # 转换为灰度图像
    image_gray_float = np.float32(image_gray) # 转换为浮点数

    # 傅里叶变换
    image_dft = cv.dft(image_gray_float, flags=cv.DFT_COMPLEX_OUTPUT)
    image_dft_shift = np.fft.fftshift(image_dft)  # 中心化

    # 计算幅值谱
    magnitude_spectrum = 20 * np.log(cv.magnitude(image_dft_shift[:, :, 0], image_dft_shift[:, :, 1]))

    # 显示幅值谱
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 4, 1)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum')
    plt.xticks([]), plt.yticks([])

    if sub_option == '低通滤波':
        cutoff_frequency_lp = 30
        filtered_img = apply_low_pass_filter(image_gray_float, cutoff_frequency_lp)

    elif sub_option == '高通滤波':
        cutoff_frequency_hp = 30
        filtered_img = apply_high_pass_filter(image_gray_float, cutoff_frequency_hp)

    elif sub_option == '带阻(陷波)滤波':
        low_cutoff_bs = 10
        high_cutoff_bs = 50
        filtered_img = apply_band_stop_filter(image_gray_float, low_cutoff_bs, high_cutoff_bs)

    
    return filtered_img

