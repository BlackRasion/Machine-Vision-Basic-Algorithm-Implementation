import cv2 as cv
import numpy as np

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