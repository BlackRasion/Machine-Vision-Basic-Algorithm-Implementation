import numpy as np
import cv2 as cv
# 区域生长
def region_growing(image, seed_point, threshold):
    """
    区域生长算法，基于给定的种子点和相似性阈值对图像进行分割。
    
    参数:
    - image: 输入的灰度图像（numpy数组）。
    - seed_point: 表示种子点坐标的元组 (x, y)。
    - threshold: 像素包含在区域内的相似性阈值。
    
    返回:
    - segmented_image: 分割后的图像，其中区域标记为255。
    """
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # 转换为灰度图像
    image = np.float32(image)

    # 获取图像的尺寸
    rows, cols = image.shape
    
    # 创建一个掩膜以跟踪访问过的像素
    visited = np.zeros_like(image, dtype=bool)
    
    # 初始化分割图像为零
    segmented_image = np.zeros_like(image)
    
    # 使用栈进行洪水填充
    stack = [seed_point]
    
    # 标记种子点为已访问并属于区域的一部分
    visited[seed_point] = True
    segmented_image[seed_point] = 255
    
    while stack:
        current_point = stack.pop()
        
        # 获取当前点的强度
        current_intensity = image[current_point]
        
        # 定义邻居（4连通性：上、下、左、右）
        neighbors = [(current_point[0] + 1, current_point[1]),
                     (current_point[0] - 1, current_point[1]),
                     (current_point[0], current_point[1] + 1),
                     (current_point[0], current_point[1] - 1)]
        
        for neighbor in neighbors:
            x, y = neighbor
            
            # 检查邻居是否在边界内且未被访问过
            if 0 <= x < rows and 0 <= y < cols and not visited[x, y]:
                neighbor_intensity = image[x, y]
                
                # 检查邻居的强度是否在阈值范围内
                if abs(neighbor_intensity - current_intensity) <= threshold:
                    stack.append((x, y))
                    visited[x, y] = True
                    segmented_image[x, y] = 255
    
    return segmented_image