# -*- coding: utf-8 -*-
import json
import numpy as np
import heapq


def smoothDistance(list):
    # 参数

    smooth_x_list = [x - theta_x for x in list]

    for i in range(1, len(list)):
        x0 = smooth_x_list[i - 1]
        x1 = smooth_x_list[i]

        if abs(x1 - x0) >= max_threshold_x:
            # 这里是异常点
            if x1 >= x0:
                x1 = beta_x * x0 + (1 - beta_x) * (x0 + alpha_x)
            else:
                x1 = beta_x * x0 + (1 - beta_x) * (x0 - alpha_x)
        else:
            # 正常点
            x1 = beta_x * x0 + (1 - beta_x) * x1

        smooth_x_list[i] = x1

    return smooth_x_list


def smoothVelocity(smooth_x_list):
    smooth_v_list = []

    smooth_v_list.append((smooth_x_list[1] - smooth_x_list[0]) / float(float(time_list[1]) - float(time_list[0])))
    for i in range(1, len(smooth_x_list)):
        smooth_v_list.append(
            (smooth_x_list[i] - smooth_x_list[i - 1]) / float(float(time_list[i]) - float(time_list[i - 1])))

    for i in range(1, len(smooth_v_list)):
        x0 = smooth_v_list[i - 1]
        x1 = smooth_v_list[i]

        if abs(x1 - x0) >= max_threshold_v:
            # 这里是异常点
            if x1 >= x0:
                x1 = beta_v * x0 + (1 - beta_v) * (x0 + alpha_v)
            else:
                x1 = beta_v * x0 + (1 - beta_v) * (x0 - alpha_v)
        else:
            # 正常点
            x1 = beta_v * x0 + (1 - beta_v) * x1

        smooth_v_list[i] = x1

    return smooth_v_list


def caculate_loss(x_s, real_x_s, smooth_x_s, v_s, real_v_s, smooth_v_s):
    # 转换成numpy数组
    x_s_np = np.array(x_s)
    real_x_s_np = np.array(real_x_s)
    smooth_x_s_np = np.array(smooth_x_s)
    v_s_np = np.array(v_s)
    real_v_s_np = np.array(real_v_s)
    smooth_v_s_np = np.array(smooth_v_s)

    # 计算loss值
    # loss_x_raw = sum(map(abs, (x_s_np - real_x_s_np)/real_x_s_np))/len(real_x_s_np)
    loss_x_smooth = sum(map(abs, (smooth_x_s_np - real_x_s_np) / real_x_s_np)) / len(real_x_s_np)
    # loss_v_raw = sum(map(abs, (v_s_np - real_v_s_np)))/len(real_v_s_np)
    loss_v_smooth = sum(map(abs, (smooth_v_s_np - real_v_s_np))) / len(real_v_s_np)

    # print("loss_x_raw: " + str(loss_x_raw))
    # print("loss_x_smooth: " + str(loss_x_smooth))
    # print("loss_v_raw: " + str(loss_v_raw))
    print("loss_v_smooth: " + str(loss_x_smooth))
    print('-------beta_x == ' + str(beta_x) + '---------------number_i == ' + str(number_i) + '---------------')
    para_arr[video_index][para_index] = loss_x_smooth
    # print(beta_v_arr[video_index][beta_v_index])


def testVideo(pre_list, real_list, time_list, faster_list, start, end, number_i):
    v_s = []
    x_s = []
    a_s = []
    real_x_s = []
    real_v_s = []
    real_a_s = []
    faster_x_s = []
    faster_v_s = []

    a_s.append(3)

    for i in range(0, len(pre_list)):
        pre_object = pre_list[i]
        real_object = real_list[i]
        faster_object = faster_list[i]
        vx = pre_object['vx']
        x = pre_object['x']

        real_vx = real_object['vx']
        real_x = real_object['x']

        faster_x = faster_object['vx']
        faster_v = faster_object['x']

        v_s.append(vx)
        x_s.append(x)

        faster_x_s.append(faster_x)
        faster_v_s.append(faster_v)

        real_x_s.append(real_x)
        real_v_s.append(real_vx)

    smooth_x_s = smoothDistance(x_s)
    smooth_v_s = smoothVelocity(smooth_x_s)

    # frames = [i for i in range(start, end)]
    # # 绘制距离图
    # plt.figure(figsize=(8, 4))
    # # # X轴，Y轴数据
    # x = frames[start: end]
    # y = x_s[start: end]
    # pltFigure(x, y, 'predict_distance', 'r--', 'frame', 'distance', 'Distance', 1, 0, len(x), 0.0, min(y), max(y))
    # y = real_x_s[start: end]
    # pltFigure(x, y, 'real_distance', 'g--', 'frame', 'distance', 'Distance', 1, 0, len(x), 0.0, min(y), max(y))
    # y = smooth_x_s[start: end]
    # pltFigure(x, y, 'smooth_distance', 'b--', 'frame', 'distance', 'Distance', 1, 0, len(x), 0.0, min(y), max(y))
    # # y = faster_x_s[start: end]
    # # pltFigure(x, y, 'faster_distance', 'y--', 'frame', 'distance', 'Distance', 1, 0, len(x), 0.0, min(y), max(y))
    #
    # plt.show()
    #
    #
    #
    # with open(pre_filepath) as f:
    #     pre_video = json.load(f)
    #
    # plt.figure(figsize=(8, 4))
    # # X轴，Y轴数据
    # x = frames[start: end]
    # #y = v_s[start: end]
    # #pltFigure(x, y, 'predict_velovity', 'r--', 'frame', 'velovity', 'Distance', 1, 0, len(x), 0.0, min(y), max(y))
    # y = real_v_s[start: end]
    # pltFigure(x, y, 'real_velovity', 'g--', 'frame', 'velovity', 'Distance', 1, 0, len(x), 0.0, min(y), max(y))
    # y = smooth_v_s[start: end]
    # pltFigure(x, y, 'smooth_velovity', 'b--', 'frame', 'velovity', 'Distance', 1, 0, len(x), 0.0, min(y), max(y))
    # # y = faster_v_s[start: end]
    # # pltFigure(x, y, 'avg_velovity', 'y--', 'frame', 'velovity', 'Distance', 1, 0, len(x), 0.0, min(y), max(y))
    # # y = smooth_v_s_pre[start: end]
    # # pltFigure(x, y, 'smooth_v_s_pre', 'k--', 'frame', 'velovity', 'Distance', 1, 0, len(x), 0.0, min(y), max(y))
    # plt.show()

    caculate_loss(x_s, real_x_s, smooth_x_s, v_s, real_v_s, smooth_v_s)


def find_best_parameter(para_arr):
    # 找出最佳的参数，求取参数数组的纵向平均值
    avg_loss = np.mean(para_arr, axis=0)
    # 求取avg_loss的最小值
    min_index = heapq.nsmallest(1, range(len(avg_loss)), avg_loss.take)


def getFrameGap(time_gap_times):
    time_list = []
    time_f = open(time_gap_times)
    while True:
        line = time_f.readline().strip('\n')
        if not line:
            break
        time_list.append(line)
    time_f.close()
    return time_list


if __name__ == "__main__":
    # 参数初始化------parameter setting------

    # 距离平滑相关超参数（beta_x，theta_x，alpha_x，max_threshold_x）具体公式见smoothDistance函数 distance X parameter
    t = 0.05
    beta_x = 0.6
    theta_x = 1.0
    alpha_x = 0.2
    # max_threshold_x = max_realtive_v * t + 2 * max_absolute_a * t * t
    max_threshold_x = 0.65

    # 速度平滑超参数（beta_v，max_threshold_v，alpha_v）具体公式见smoothVelocity函数 velocity V parameter
    beta_v = 0.7
    max_threshold_v = 0.8
    alpha_v = 0.2

    # 视频列表，01是视频的尾号
    number_val = ['01', '02', '03']

    # 参数的测试范围（供参考），飘依，你可以自己设定参数变化步长
    video_index = 0

    beta_v_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    max_threshold_v_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    alpha_v_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

    beta_x_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    max_threshold_x_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    alpha_x_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    theta_x_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                    2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]

    # 采用控制变量法来尝试找到最优参数，但是参数之间相互影响，不建议再次采用这种方式
    # 推荐按照昨天帅楠说的方式（多重for循环，寻找最优的搭配）
    para_list = beta_x_list
    para_arr = np.arange(len(para_list) * len(number_val)).reshape(len(number_val), len(para_list))
    para_arr = para_arr.astype(float)
    print(para_arr.dtype)

    # 读取视频文件
    for number_i in number_val:
        para_index = 0
        pre_filepath = 'C:/Users/lirunze/MCDC/data/data_02/faster/myRawJson_valid_' + number_i + '.json'
        real_filepath = 'C:/Users/lirunze/MCDC/data/data_02/gt/valid_video_' + number_i + '_gt.json'

        with open(pre_filepath) as f:
            pre_video = json.load(f)
        with open(real_filepath) as f:
            real_video = json.load(f)

        pre_list = pre_video['frame_data']
        real_list = real_video['frame_data']

        time_list = getFrameGap('C:/Users/lirunze/MCDC/data/train/valid/valid_video_00_time.txt')
        for beta_x in para_list:
            testVideo(pre_list, real_list, time_list, real_list, 0, 300, para_index)
            para_index += 1
        video_index += 1

    find_best_parameter(para_arr)
