import numpy as np
import cv2 as cv

def My_template_match(img, template):
    """
    函数功能：
        实现模板匹配
        使用最简单的平方差(SSD, Sum of Squared Differences)作为相似度度量,
        即cv2.TM_SQDIFF方法的等效实现

    return: 
        min_loc (最小位置), 最佳匹配位置的左上角坐标
        min_val (最小值), 最小的平方差(SSD)值
    """
    img_height, img_width = img.shape
    temp_height, temp_width = template.shape

    # 初始化最小SSD值为无穷大
    min_ssd = float('inf')
    best_position = (0, 0)

    # 遍历所有可能的位置
    for y in range(img_height - temp_height + 1):
        for x in range(img_width - temp_width + 1):
            # 提取子区域
            sub_image = img[y:y+temp_height, x:x+temp_width]

            # 计算SSD
            ssd = np.sum((sub_image.astype("float") - template.astype("float")) ** 2)

            # 更新最小SSD值和最佳位置
            if ssd < min_ssd:
                min_ssd = ssd
                best_position = (x, y)

    return best_position, min_ssd

def template_match(img, sub_option):
    result_img = img.copy() # 复制原始图像
    result_img = cv.cvtColor(result_img, cv.COLOR_BGR2GRAY) # 将彩色图像转换为灰度图像

    template = cv.imread('fish.png', 0) # 加载灰度模板图像
    h, w = template.shape[:2] # 获取模板图像的高和宽

    # 选择匹配方法
    if sub_option == '单对象匹配':
        res = cv.matchTemplate(result_img, template, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        left_top = max_loc # 左上角坐标
        right_bottom = (left_top[0] + w, left_top[1] + h) # 右下角坐标
        cv.rectangle(result_img, left_top, right_bottom, 255, 1) # 绘制矩形
    elif sub_option == '多对象匹配':
        res = cv.matchTemplate(result_img, template, cv.TM_CCOEFF_NORMED)
        threshold = 0.8 # 设置阈值
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            cv.rectangle(result_img, pt, (pt[0] + w, pt[1] + h), 255, 1)
    elif sub_option == '自设模板匹配':
        left_top, min_val = My_template_match(result_img, template)
        right_bottom = (left_top[0] + w, left_top[1] + h)
        cv.rectangle(result_img, left_top, right_bottom, 255, 1)
    return result_img







