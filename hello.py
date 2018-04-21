# -*- encoding:utf-8 -*-
import random
import math
import colorsys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import bird_view_projection as birdView
import json
import CONFIG_SERVER_TEST as CONFIG
import smooth as smooth
import infer_simple as infer

# box colors
box_colors = None


def generate_colors(num_classes):
    global box_colors

    if box_colors != None and len(box_colors) > num_classes:
        return box_colors

    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    box_colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    box_colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            box_colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    # Shuffle colors to decorrelate adjacent classes.
    random.shuffle(box_colors)
    random.seed(None)  # Reset seed to default.


def draw_boxes(img, result):

    image = Image.fromarray(img)

    font = ImageFont.truetype(str.encode(CONFIG.DARKNET_DIR+'font/FiraMono-Medium.otf'), 20)
    thickness = (image.size[0] + image.size[1]) // 300

    num_classes = len(result)
    generate_colors(num_classes)

    index = 0
    for objection in result:
        index += 1
        class_name, class_score, (x, y, w, h) = objection
        # print(name, score, x, y, w, h)

        left = int(x - w / 2)
        right = int(x + w / 2)
        top = int(y - h / 2)
        bottom = int(y + h / 2)

        label = '{} {:.2f}'.format(class_name.decode('utf-8'), class_score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        #print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i,
                            bottom - i], outline=box_colors[index - 1])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=box_colors[index - 1])
        draw.text(text_origin, label, fill=(255, 255, 255), font=font)
        del draw

    return np.array(image)


def drawBoxOnImg(img,x,y,w,h,p_x,p_y,num):
    cv2.rectangle(img,(int(x),int(y)),(int(x+w),int(y+h)),(127,255,0),5)
    cv2.circle(img, (int(p_x),int(p_y)), 5, (255,0,0),-1) 
    #img_path = CONFIG.DARKNET_DIR+'pic/'+str(num)+'.jpg'
    img_path = '/data/workspace/fbdet/test_pic_test/'+str(num) +'.jpg'
    cv2.imwrite(img_path,img, [int( cv2.IMWRITE_JPEG_QUALITY), 20])


def keep_img(img,num):
    # 在valid上面做测试  先保存图片，下载自己用
    img_path = '/data/workspace/fbdet/test_pic/'+str(num) +'.jpg'
    cv2.imwrite(img_path,img, [int( cv2.IMWRITE_JPEG_QUALITY), 20])


def chooseOnImprove2(pic_list, cam):
    # remove self car
    if pic_list is None:
        return None
    width = float(cam['image_width'])
    height= float(cam['image_height'])
    left = float(cam['cam_to_right'])
    right= float(cam['cam_to_left'])
    x_car_mid = (width * left / (left + right) + width / 2) / 2  
    # x_car_mid=width/2
    # x_car_mid= width*left/(left+right)/5 +width*2/5

    i = 0
    while i < len(pic_list):
        iterater = pic_list[i]
        # print(iterater)
        p0 = iterater[2][0]
        p1 = iterater[2][1]
        w = iterater[2][2]
        h = iterater[2][3]

        if not (iterater[0] == b'car' or iterater[0] == b'truck' or iterater[0] == b'bus'):
            pic_list.remove(iterater)
            continue
        # elif h / w > 1.4:
        #     pic_list.remove(iterater)
        #     continue
        elif abs(x_car_mid - p0) > width / 5:
            pic_list.remove(iterater)
            continue
        # elif p1 > height * 0.9 and w > width*0.8:
        elif p1+h/2 > height * 0.92 and w > width*0.7 and h < height * 0.4:
            pic_list.remove(iterater)
            continue
        i = i + 1

    if len(pic_list) == 0:
        return None
    pic_list = sorted(pic_list, key=lambda x: -x[2][1])
    # print('----------------------choose', pic_list[0])
    return pic_list[0]

def chooseOneImproveWithTracking(pic_list, cam, pre__x, pre__y, pre__w):
    if pic_list is None:
        return None  
    width = float(cam['image_width'])
    height= float(cam['image_height'])
    left = float(cam['cam_to_right'])
    right= float(cam['cam_to_left'])
    x_car_mid = (width * left / (left + right) + width / 2) / 2  

    i = 0
    while i < len(pic_list):
        iterater = pic_list[i]
        # print(iterater)
        p0 = iterater[2][0]
        p1 = iterater[2][1]
        w = iterater[2][2]
        h = iterater[2][3]

        dist2 = math.sqrt((p0-w/2-pre__x)*(p0-w/2-pre__x)+(p1-h/2-pre__y)*(p1-h/2-pre__y))
        #print( str(dist2)+" "+str(pre__w/3))
        if not (iterater[0] == u'car' or iterater[0] == u'truck' or iterater[0] == u'bus'):
            pic_list.remove(iterater)
            continue
        elif dist2 > pre__w/3:
            pic_list.remove(iterater)
            continue
        elif abs(x_car_mid - p0) > width / 3:
            pic_list.remove(iterater)
            continue
        # elif p1 > height * 0.9 and w > width*0.8:
        elif p1+h/2 > height * 0.92 and w > width*0.7 and h < height * 0.4:
            pic_list.remove(iterater)
            continue
        i = i + 1

    if len(pic_list) == 0:
        return None  

    pic_list = sorted(pic_list, key=lambda x: -x[2][1])
    # print('----------------------choose', pic_list[0])
    return pic_list[0]


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


def changeCoordinate(bboxes):
    # [ car, x,y,w,h] -> ( car, 0, (x_mid,y_mid,w,h) )
    res = []
    for i in range(len(bboxes)):
        res.append((bboxes[i][0], 0.9,
                    (bboxes[i][1] + bboxes[i][3] / 2, bboxes[i][2] + bboxes[i][4] / 2, bboxes[i][3], bboxes[i][4])))
    return res


frameId = 0


def handleVideo(video_path, time_txt_name, output_result_json_path, camera_param_json_name):
    global frameId
    video = cv2.VideoCapture(video_path)
    # print('------------open video:',video_path)
    # # Find OpenCV version
    # (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    #
    # if int(major_ver)  < 3 :
    #     fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    #     print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    # else :
    #     fps = video.get(cv2.CAP_PROP_FPS)
    #     print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    birdView.setCameraParams(camera_param_json_name)

    with open(camera_param_json_name, 'r') as f:
        temp = json.loads(f.read())

    count_frame, process_every_n_frame = 0, 1

    pre_x = 960.00
    pre_y = 960.00
    pre_dis_x = 0

    time_list = getFrameGap(time_txt_name)

    #print(time_list)
    #print('------------read time_txt finished, lines:', len(time_list),time_txt_name)

    result_list = []

    car_list = ['/Users/wangshuainan/Desktop/image/1523465188473.jpg',
                '/Users/wangshuainan/Desktop/image/1523465217730.jpg',
                '/Users/wangshuainan/Desktop/image/1523465247087.jpg']
    i = 0

    while (True):
        #print( str(pre_box_x)+" "+str(pre_box_y)+" "+str(pre_box_w))
        # if frameid % 100 == 0:
        #     print('-----------------------frameid:', frameid)
        ret, img = video.read()
        # if i < 3:
        #     img = cv2.imread(car_list[i])
        #     i += 1
        # else:
        #     break
        # if img is None:
        #     print("video.read() fail || video.read() is end!")
        #     break
        if img is None or ret is None:
            print("video.read() fail || video.read() is end!")
            break

        # show a frame
        # img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # resize image half
        # cv2.imshow("Video", img)

        if count_frame % process_every_n_frame == 0:
            boxes = infer.pipeline_mask(img)
            boxex_mid_point = changeCoordinate(boxes)
            if i > 0:
                only_box = chooseOneImproveWithTracking(boxex_mid_point, temp, pre_box_x, pre_box_y, pre_box_w)
            else:
                only_box = chooseOnImprove2(boxex_mid_point, temp)

            if i > 0 and only_box is None:
                only_box = chooseOnImprove2(boxex_mid_point, temp)

            # 如果定位框返回空的话，用前面的框
            if only_box is not None:
                # return a tuple : (b'truck', 0.9237195253372192, (581.048583984375, 128.2719268798828, 215.67906188964844, 85.07489776611328))
                class_name, class_score, (x, y, w, h) = only_box
                # print(name, score, x, y, w, h)/home/m1/Downloads/optimize_parameter.py
                left = int(x - w / 2)
                right = int(x + w / 2)
                top = int(y - h / 2)
                bottom = int(y + h / 2)

                x1, y1 = x, y + int(h/2)

                box_x = int(x - w / 2)
                box_y = int(y - h / 2)
                box_w = w
                box_h = h

                pre_x, pre_y = x1, y1
                pre_box_x, pre_box_y, pre_box_w, pre_box_h = box_x, box_y, box_w, box_h
                #print('------------------only_box is Not null:', x1, y1)
             
            else:
                x1, y1 = pre_x, pre_y
                box_x, box_y, box_w, box_h = pre_box_x, pre_box_y, pre_box_w, pre_box_h
                #print('the '+str(i)+' pic------------------only_box is null', x1, y1)

            #drawBoxOnImg(img, box_x, box_y, box_w, box_h, x1, y1, frameId)

            # 然后计算速度+距离
            # distance_x代表相距前车距离
            #print('--------------------------birdView.getXY---')
            distance_x, distance_y = birdView.getXY(x1, y1)
            #print('--------------------------birdView.getXY---',distance_x, distance_y)
            if count_frame > 0:
                speed_x = (distance_x - pre_dis_x) / float(
                    float(time_list[count_frame]) - float(time_list[count_frame - 1]))
            else:
                # 第一帧的速度默认为10m/s,然后最后输出时再用第二帧的速度去校正它
                speed_x = 10
            pre_dis_x = distance_x
            pre_speed_x = speed_x

            # test_video_00_pre.json
            #  {
            #  "vx": -2.3125, //相对速度
            #  "x": 11.0625, //相对位置
            #  "fid": 0 //frame_id, 帧号，输出时帧号从 0 开始顺序依次递增
            #  }
            # }
            dict = {'vx': speed_x, 'x': distance_x, "fid": i, 
                    'ref_bbox': {"top": box_y, "right": box_x+box_w,
                                 "bot": box_y+box_h, "left": box_x}}

            result_list.append(dict)
            count_frame += 1
            i += 1

            frameId += 1

    smooth_result_list = smooth.smoothData(result_list,time_list)        
    # print('=========pipeline finished,result============>')
    # print(smooth_result_list)
    # print('=============================================>')

    tmp_dict = {'frame_data': result_list}
    # DO YOUR JSON CONV JOB!!!
    final_dict = {'frame_data': smooth_result_list}

    with open(output_result_json_path,'w+') as json_file:
        json.dump(final_dict, json_file, ensure_ascii = False)

    # row data to file:
    with open(output_result_json_path[0:-5]+'_raw.json','w+') as row_json_file:
        json.dump(tmp_dict, row_json_file, ensure_ascii = False)

    print('======pipeline finished,write json:'+output_result_json_path+' finished======>')
    video.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":

    with open('/data/workspace/fbdet/camera_parameter.json', 'r') as f:
        temp = json.loads(f.read())

    car_list = ['/data/workspace/fbdet/test_pic/00.jpg',
                '/data/workspace/fbdet/test_pic/11.jpg',
                '/data/workspace/fbdet/test_pic/22.jpg']
    i=0
    while i<len(car_list):
        img = cv2.imread(car_list[i])

        boxes = infer.pipeline_mask(img)
        boxex_mid_point = changeCoordinate(boxes)
        if i > 0:
            only_box = chooseOneImproveWithTracking(boxex_mid_point, temp, pre_box_x, pre_box_y, pre_box_w)
        else:
            only_box = chooseOnImprove2(boxex_mid_point, temp)

        class_name, class_score, (x, y, w, h) = only_box
        # print(name, score, x, y, w, h)
        left = int(x - w / 2)
        right = int(x + w / 2)
        top = int(y - h / 2)
        bottom = int(y + h / 2)

        x1, y1 = x, y + int(h/2)

        box_x = int(x - w / 2)
        box_y = int(y - h / 2)
        box_w = w
        box_h = h

        drawBoxOnImg(img, box_x, box_y, box_w, box_h, x1, y1, i+3)

        i += 1

