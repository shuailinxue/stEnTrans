import cv2
import numpy as np

def nedi(input, new_dimension, side_2, threshold):
    (m, n) = input.shape
    (N, M) = new_dimension
    img_var = np.zeros((m, n))
    output = cv2.resize(input, new_dimension, interpolation=cv2.INTER_LINEAR)  # 先双线性插值算法
    for i in range(m):  # 遍历图中所有像素
        for j in range(n):

            dot = input[i, j]  # 框定像素点的领域窗口
            dot_x1 = max(0, i - side_2)
            dot_x2 = min(i + side_2 + 1, n)
            dot_y1 = max(0, j - side_2)
            dot_y2 = min(j + side_2 + 1, n)
            dot_var = np.var(input[dot_x1:dot_x2, dot_y1:dot_y2])  # 计算窗口内的方差
            img_var[i, j] = dot_var  # 统计各点的方差，方便阈值的调整
            if dot_var > threshold:  # 判断点的方差是否是边缘点
                output[int(i * M / m - 0.5), int(j * N / n - 0.5)] = dot
    return output, img_var  # 输出图像与方差图

def NEDI_run(img, new_size, side_2=1, threshold=0.5, ):
    NEDI_img, NEDI_var = nedi(img, new_size, side_2, threshold)
    return NEDI_img