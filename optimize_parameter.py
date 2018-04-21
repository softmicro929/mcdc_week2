# -*- coding: utf-8 -*-
import json
import numpy as np
import heapq
#from matplotlib import pyplot as plt


dist= [.0]*30
cnt = [0]*30

def smoothDistance(list_x, beta_x, theta_x, alpha_x, max_threshold_x):
    # 参数
    smooth_x_list = [x - theta_x for x in list_x]

    for i in range(1, len(list_x)):
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

    # frames = [i for i in range(0, 499)]
    # # 绘制距离图
    # plt.figure(figsize=(8, 4))
    # # # X轴，Y轴数据
    # x = frames[start: end]
    # y = list_x[start: end]
    # pltFigure(x, y, 'predict_distance', 'r--', 'frame', 'distance', 'Distance', 1, 0, len(x), 0.0, min(y), max(y))
    # # y = real_x_s[start: end]
    # # pltFigure(x, y, 'real_distance', 'g--', 'frame', 'distance', 'Distance', 1, 0, len(x), 0.0, min(y), max(y))
    # y = smooth_x_list[start: end]
    # pltFigure(x, y, 'smooth_distance', 'b--', 'frame', 'distance', 'Distance', 1, 0, len(x), 0.0, min(y), max(y))
    # # y = faster_x_s[start: end]
    # # pltFigure(x, y, 'faster_distance', 'y--', 'frame', 'distance', 'Distance', 1, 0, len(x), 0.0, min(y), max(y))
    #
    # plt.show()


    return smooth_x_list


# def smoothVelocity(smooth_x_list):
#     smooth_v_list = []
#
#     smooth_v_list.append((smooth_x_list[1] - smooth_x_list[0]) / float(float(time_list[1]) - float(time_list[0])))
#     for i in range(1, len(smooth_x_list)):
#         smooth_v_list.append(
#             (smooth_x_list[i] - smooth_x_list[i - 1]) / float(float(time_list[i]) - float(time_list[i - 1])))
#
#     for i in range(1, len(smooth_v_list)):
#         x0 = smooth_v_list[i - 1]
#         x1 = smooth_v_list[i]
#
#         if abs(x1 - x0) >= max_threshold_v:
#             # 这里是异常点
#             if x1 >= x0:
#                 x1 = beta_v * x0 + (1 - beta_v) * (x0 + alpha_v)
#             else:
#                 x1 = beta_v * x0 + (1 - beta_v) * (x0 - alpha_v)
#         else:
#             # 正常点
#             x1 = beta_v * x0 + (1 - beta_v) * x1
#
#         smooth_v_list[i] = x1
#
#     return smooth_v_list

#######################################################################################################


def caculate_loss_x( real_x_s, smooth_x_s ):
    # 转换成numpy数组
    real_x_s_np = np.array(real_x_s)
    smooth_x_s_np = np.array(smooth_x_s)
    # 计算loss值
    loss_x_smooth = sum(map(abs, (smooth_x_s_np - real_x_s_np) / real_x_s_np)) / len(real_x_s_np)
    return loss_x_smooth
    # print(beta_v_arr[video_index][beta_v_index])
def caculate_loss_v( v_s, real_v_s, smooth_v_s):
    # 初始速度怎么来的？。。
    v_s_np = np.array(v_s)
    real_v_s_np = np.array(real_v_s)
    smooth_v_s_np = np.array(smooth_v_s)

    loss_v_smooth = sum(map(abs, (smooth_v_s_np - real_v_s_np))) / len(real_v_s_np)

    return loss_v_smooth
    # print(beta_v_arr[video_index][beta_v_index])




def calcLoss_x(beta_x, theta_x, alpha_x, max_threshold_x, pre_list, real_list, time_list):
    # 平滑处理某个视频的x， 和gt的计算loss
    global cnt
    global dist
    # 预测是x  x和gt的差值+上去
    res_calcLoss= 0.0
    list_x=[]
    real_list_x=[]
    for i in range(len(pre_list)):
        tp = pre_list[i]
        list_x.append(tp[u'x'])
        tp2 = real_list[i]
        real_list_x.append(tp2[u'x'])

        aha = tp[u'x']
        pos = int(round(aha))
        cnt[pos] += 1
        dist[pos] += aha-tp2[u'x']
        # 之后减去这个数


        #print( str(tp['x']) + '  '+ str(tp2['x']))

    #smooth_x = smoothDistance(list_x, beta_x, theta_x, alpha_x, max_threshold_x)

    #print(smooth_x)
    # 500 个相对差
    #x_s_np = np.array(smooth_x)
    x_s_np = np.array(list_x)
    real_x_s_np = np.array(real_list_x)

    # 计算loss值
    res_calcLoss = sum(map(abs, (x_s_np - real_x_s_np)/real_x_s_np))/len(real_x_s_np)
    return res_calcLoss




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

def calcALLVideoLoss_x(beta_x,theta_x,alpha_x,max_threshold_x):
    # 计算这些参数在4个raw_json的损失值
    res_loss_x=0.0
    number_val = ['00','01','02','03','04','05' ]

    for number_i in number_val:

        #pre_filepath = '/home/m1/Downloads/4.20valid_data/res_json/valid_video_' + number_i + '_pre_raw.json'
        pre_filepath = '/home/m1/Downloads/4.20valid_linearregression/mcdc/valid_video_' + number_i + '_pre.json'
        real_filepath = '/home/m1/Downloads/4.20valid_video/valid/valid_video_' + number_i + '_gt.json'
        time_list = getFrameGap('/home/m1/Downloads/4.20valid_video/valid/valid_video_'+number_i+'_time.txt')
        #print('handle file:'+ pre_filepath)
        with open(pre_filepath) as f:
            pre_video = json.load(f)
        with open(real_filepath) as f:
            real_video = json.load(f)

        pre_list = pre_video['frame_data']
        real_list = real_video['frame_data']
        # 这里还是元组list
        # print( pre_list)
        # print( real_list)
        res_loss_x += calcLoss_x(beta_x, theta_x, alpha_x, max_threshold_x, pre_list, real_list, time_list)


    return res_loss_x/len(number_val)



if __name__ == "__main__":

    # beta_v_list = [0.4, 0.6, 0.8]
    # max_threshold_v_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    # alpha_v_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]




    #beta_x_list = [  0.5,0.7,0.9]  #权重
    # beta_x_list = [ 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]  #权重
    #
    # theta_x_list = [ -1,-0.7,-0.4,-0.1,0.2,0.5,0.8,1.1,1,5 ] # 总体+- 调节测量误差
    # alpha_x_list = [ 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # max_threshold_x_list = [0.2,0.4, 0.6, 0.8, 1.0, 1,5,1,8 ]

    # beta_x_list = [0.5,0.7,0.9]  #权重
    #
    # theta_x_list = [-0.4,-0.1,0.2,0.5,0.8,1.1] # 总体+- 调节测量误差
    #
    # alpha_x_list = [0.2,0.4 ]
    # max_threshold_x_list = [0.2,0.4, 0.6, 0.8, 1.0, 1,5,1,8 ]



    beta_x_list = [0.5]  #权重

    theta_x_list = [0] # 总体+- 调节测量误差

    alpha_x_list = [ 0.4 ]
    max_threshold_x_list = [ 1,8 ]


    # 0.056141631984
    # [0.8, 0.8, 0.1, 1.0]



    # 0.0562092664515   用了6个视频
    # [0.7, 0.9, 0.2, 1.0]  ????????//


    # beta_x_list = [ 0.699, 0.7, 0.701 ]
    # theta_x_list = [ 0.9, 0.91,0.92 ]
    # alpha_x_list = [ 1.04,1.041,1.042,1.043 ]
    # max_threshold_x_list = [ 0.599, 0.6, 0.601 ]
    # 0.0586799484862
    # [0.701, 0.91, 1.043, 0.599]


    # 0.0586880287988
    # [0.7, 0.9, 1.05, 0.6]


    # 0.0466834082735
    # [0.75, 0.9, 1.05, 0.65]

    # 0.0586880287988
    # [0.7, 0.9, 1.05, 0.6]


    # 0.0587214379261
    # [0.7, 0.9, 1.0, 0.5]

    # 0.0587505520438
    # [0.7, 0.9, 0.9, 0.5]


    # 0.0596658536995
    # [0.8, 0.8, 0.8, 0.4]

    # beta_v_list = [0.5, 0.6]
    # max_threshold_v_list = [0.3, 0.5]
    # alpha_v_list = [ 0.4, 0.5 ]
    #
    # beta_x_list = [ 0.6, 0.8]
    # max_threshold_x_list = [0.4 , 0.5]
    # alpha_x_list = [0.3, 0.4]
    # theta_x_list = [0.1,  1.0]

    # 取值太分散了，10的7次方 速度爆炸。。 先在大范围内寻找，然后再来一次，在小范围内寻找
# x,v 可以分开计算
    best_para_x = []
    best_para_v = []
    min_loss_x = 9.0
    min_loss_v = 9.0
    max_loss_x= 0.1

    for beta_x in beta_x_list:
        for theta_x in theta_x_list:
            for alpha_x in alpha_x_list:
                for max_threshold_x in max_threshold_x_list:
                    cur_para= [beta_x, theta_x, alpha_x, max_threshold_x]
                    temp = calcALLVideoLoss_x(beta_x,theta_x,alpha_x,max_threshold_x)
                    # print( cur_para)
                    # print( temp )
                    if temp < min_loss_x:
                        best_para_x = [beta_x,theta_x,alpha_x,max_threshold_x]
                        min_loss_x = temp
                    if temp > max_loss_x:
                        max_loss_x = temp

    print(max_loss_x)
    print(min_loss_x)
    print(best_para_x)

    # print(dist)
    # print (cnt)
    # for i in range(0, len(cnt)):
    #     if cnt[i] != 0:
    #         dist[i] = dist[i]/cnt[i]
    #
    # print(dist)



# 00
# 0.11630113032
# 0.0321213510722
# [0.9, -0.4, 0.2, 0.4]
# 我觉得 微信。。。
# 01
# 0.170921665357
# 0.0349458749977
# [0.5, 1.1, 0.2, 0.6]
#
# 02
# 0.16410484339
# 0.0280846613366
# [0.7, 1.1, 0.2, 0.4]
#
# 03
# 0.238540157155
# 0.0356918626964
# [0.5, 1.1, 0.4, 0.2]
#
# 04
# 0.1
# 0.0336010300183
# [0.7, -0.4, 0.2, 0.4]
#
# 05
# 0.104953783914
# 0.0346047951093
# [0.9, 0.5, 0.2, 5]





[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 150.26855799999998, 82.92697799999998, 657.9067199999996, 840.7573340000009, 1094.1796339691157, 458.6376700617671, 95.40917593823237, 87.43671596911624, -245.4832360000001, -74.31815999999986, -160.39339199999998, -111.01515400000004, -257.84378999999996, -372.9122700000001, -60.187774000000005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[0, 0, 0, 0, 0, 0, 112, 70, 576, 884, 1128, 626, 282, 252, 602, 546, 334, 170, 146, 238, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 1.3416835535714284, 1.184671114285714, 1.142199166666666, 0.9510829570135757,
 0.9700174060009891, 0.7326480352424395, 0.33833041112848355, 0.34697109511554064, -0.4077794617940201,
 -0.1361138461538459, -0.48021973652694605, -0.6530303176470591, -1.7660533561643832, -1.5668582773109248,
 -1.7702286470588238, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0]