import cv2 as cv
import numpy as np

def pad_with(vector, pad_width, iaxis, kwargs):
    """用于np.pad的回调函数，以填充图像边界"""
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

def dilation(image, selem):
    """
    对二值图像执行膨胀操作。
    """
    # 获取结构元素的尺寸
    selem_height, selem_width = selem.shape
    selem_center = (selem_height // 2, selem_width // 2)

    # 计算需要填充的宽度
    pad_height = selem_height // 2
    pad_width = selem_width // 2

    # 填充图像以处理边界情况
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), pad_with)

    # 初始化膨胀后的图像
    dilated = np.zeros_like(image)

    # 执行膨胀操作
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # 提取与结构元素大小相同的区域
            region = padded_image[i:i+selem_height, j:j+selem_width]
            # 如果结构元素与区域的按位与结果中存在前景像素，则该位置为前景
            if np.any(selem * region):
                dilated[i, j] = 1

    return dilated

def erosion(image, selem):
    """
    对二值图像执行腐蚀操作。
    """
    # 获取结构元素的尺寸
    selem_height, selem_width = selem.shape
    selem_center = (selem_height // 2, selem_width // 2)

    # 计算需要填充的宽度
    pad_height = selem_height // 2
    pad_width = selem_width // 2

    # 填充图像以处理边界情况
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), pad_with)

    # 初始化腐蚀后的图像
    eroded = np.zeros_like(image)

    # 执行腐蚀操作
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # 提取与结构元素大小相同的区域
            region = padded_image[i:i+selem_height, j:j+selem_width]
            # 如果结构元素与区域的按位与结果全部为前景像素，则该位置为前景
            if np.all(selem <= region):
                eroded[i, j] = 1

    return eroded

def close_image(image, selem):
    """
    对二值图像执行闭运算。
    参数:
        image: 输入的二值图像 (numpy array)，其中0表示背景，1表示前景。
        selem: 结构元素 (numpy array)，用于定义邻域形状。
    
    返回:
        closed: 闭运算后的图像 (numpy array)。
    """
    # 先进行膨胀操作
    dilated = dilation(image, selem)
    # 然后进行腐蚀操作
    closed = erosion(dilated, selem)
    return closed

def morphology_process(image, sub_option):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # 转换为灰度图像
    _, dog_thresh = cv.threshold(image_gray, 150, 255, cv.THRESH_BINARY) # 二值化
    

    if sub_option == '膨胀':
        kernel = np.ones((5, 5), np.uint8) # 5x5的卷积核
        dog_morphology = cv.dilate(dog_thresh, kernel, iterations=1)
    elif sub_option == '腐蚀':
        kernel = np.ones((5, 5), np.uint8) # 5x5的卷积核
        dog_morphology = cv.erode(dog_thresh, kernel, iterations=1) # 腐蚀
    elif sub_option == '开运算':
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        dog_morphology = cv.morphologyEx(dog_thresh, cv.MORPH_OPEN, kernel) # 开运算
    elif sub_option == '闭运算':
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        dog_morphology = cv.morphologyEx(dog_thresh, cv.MORPH_CLOSE, kernel)
    elif sub_option == '形态学梯度':
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        dog_morphology = cv.morphologyEx(dog_thresh, cv.MORPH_GRADIENT, kernel)
    elif sub_option == '顶帽':
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        dog_morphology = cv.morphologyEx(dog_thresh, cv.MORPH_TOPHAT, kernel)
    elif sub_option == '黑帽':
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        dog_morphology = cv.morphologyEx(dog_thresh, cv.MORPH_BLACKHAT, kernel)

    elif sub_option == '自设闭运算':
        dog_morphology = close_image(dog_thresh, np.ones((5, 5), np.uint8))

    return dog_morphology
