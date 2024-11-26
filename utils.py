import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def show_img(title='image', img=None):
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def get_attributes(img):
    print('Image shape:', img.shape)
    print('Image size:', img.size)
    print('Image data type:', img.dtype)
    return img.shape

def BGR_split(img):
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    # show_img(B, 'Blue channel')
    # show_img(G, 'Green channel')
    # show_img(R, 'Red channel')
    return B, G, R

def plot_histogram(image, method=3):
    """
    绘制灰度直方图的函数，可以选择不同的方法。
    
    参数:
    image: 输入的灰度图像
    method: 选择绘制直方图的方法，可选值为 0-> 'opencv', 1-> 'numpy1', 2-> 'numpy2', 3-> 'matplotlib', 默认为3
    """
    if method == 0:
        # 使用OpenCV方法
        hist = cv.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(hist)
        plt.title('Histogram (OpenCV)')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.show()
        
    elif method == 1:
        # 使用numpy方法1
        hist, bins = np.histogram(image.ravel(), 256, [0, 256])
        plt.plot(hist)
        plt.title('Histogram (Numpy Method 1)')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.show()
        
    elif method == 2:
        # 使用numpy方法2
        hist = np.bincount(image.ravel(), minlength=256)
        plt.plot(hist)
        plt.title('Histogram (Numpy Method 2)')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.show()
        
    elif method == 3:
        # 使用matplotlib方法
        plt.hist(image.ravel(), bins=256, range=[0, 256])
        plt.title('Histogram (Matplotlib)')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.show()
        
    else:
        raise ValueError("Invalid method. Choose from 0 (opencv), 1 (numpy1), 2 (numpy2), 3 (matplotlib).")
    

def visualize_threshold_methods(image, threshold):
    """
    可视化5种不同的阈值分割方法，并返回用户选择的二值化图片。
    
    参数:
    image: 输入的灰度图像
    threshold: 阈值
    
    返回:
    选择的二值化图片
    """
    # 应用5种不同的阈值分割方法
    ret, th1 = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
    ret, th2 = cv.threshold(image, threshold, 255, cv.THRESH_BINARY_INV)
    ret, th3 = cv.threshold(image, threshold, 255, cv.THRESH_TRUNC)
    ret, th4 = cv.threshold(image, threshold, 255, cv.THRESH_TOZERO)
    ret, th5 = cv.threshold(image, threshold, 255, cv.THRESH_TOZERO_INV)

    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [image, th1, th2, th3, th4, th5]

    # 显示所有阈值分割方法的结果图
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([]) # 隐藏坐标轴

    plt.show()

    # 用户选择阈值分割方法
    choice = int(input("选择阈值分割方法 (1: BINARY, 2: BINARY_INV, 3: TRUNC, 4: TOZERO, 5: TOZERO_INV): "))
    
    if choice < 1 or choice > 5:
        raise ValueError("Invalid choice. Choose a number between 1 and 5.")
    
    return images[choice]