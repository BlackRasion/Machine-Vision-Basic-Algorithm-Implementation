import numpy as np
import cv2 as cv
def detect_shapes(image):
    # 将图像转换为灰度图像
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 应用高斯模糊以减少噪声并改进轮廓检测
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    # 使用Canny边缘检测器检测边缘
    edges = cv.Canny(blurred, 50, 150)
    
    # 查找边缘图像中的轮廓
    contours, _ = cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    shapes = []
    
    for contour in contours:
        shape = "unknown"
        # 计算轮廓的面积
        area = cv.contourArea(contour)
        # 跳过面积过小的轮廓
        if area < 10:
            continue

        epsilon = 0.02 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        
        # 根据顶点数量确定形状
        if len(approx) == 3:
            shape = "triangle" # 三角形
        elif len(approx) == 4:
            # 检查是否为正方形或矩形
            x, y, w, h = cv.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                shape = "square" # 正方形
            else:
                shape = "rectangle" # 矩形
        elif len(approx) == 5:
            shape = "pentagon" # 五边形
        elif len(approx) == 6:
            shape = "hexagon" # 六边形
        else:
            # 检查是否为圆形或椭圆
            perimeter = cv.arcLength(contour, True)
            circularity = 4 * np.pi * (area / (perimeter ** 2)) # 圆形度
        
            if circularity > 0.85:
                shape = "circle" # 圆形
            else:
                # 拟合椭圆并检查其偏心率
                ellipse = cv.fitEllipse(contour)
                major_axis = max(ellipse[1])
                minor_axis = min(ellipse[1])
                eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
                
                if eccentricity < 0.1:
                    shape = "circle" # 圆形
                elif eccentricity < 0.9:
                    shape = "ellipse" # 椭圆
        
        # 在图像上绘制轮廓并在中心位置标注形状名称
        cv.drawContours(image, [contour], -1, (0, 255, 0), 2)
        M = cv.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv.putText(image, shape, (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        shapes.append(shape)
    
    return image, shapes

def geometric_dectect(image, sub_option):
    if sub_option == '轮廓检测':
        result_image = image.copy()
        result_image, detected_shapes = detect_shapes(result_image)
        print(f"检测到的形状: {detected_shapes}")
    return result_image 