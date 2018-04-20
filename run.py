# -*- encoding:utf-8 -*-
import os
import os.path
import CONFIG_SERVER_TEST as CONFIG
import hello as h
import sys
def list_file():
    video_list=[]
    for filename in os.listdir(CONFIG.TEST_VIDEO_DIR):
        if os.path.splitext(filename)[1] == '.avi':
            if filename!= '效果仅做参考.avi':
                video_list.append(filename)
                #print(filename)

    for videoname in video_list:

        time_txt_name = CONFIG.TEST_VIDEO_DIR+videoname[0:-4]+'_time.txt'
        json_name     = CONFIG.WRITE_JSON_DIR+videoname[0:-4]+'_pre.json'
        video_path    = CONFIG.TEST_VIDEO_DIR+videoname
        h.handleVideo(video_path,time_txt_name,json_name, CONFIG.CAMERA_PARAMETER_PATH)

def dofile(videoname,id):
    time_txt_name = CONFIG.TEST_VIDEO_DIR + videoname[0:-4] + '_time.txt'
    print(videoname)
    json_name = CONFIG.WRITE_JSON_DIR + videoname[0:-4] + '_pre.json'
    video_path = CONFIG.TEST_VIDEO_DIR + videoname
    if id == '0':
        h.handleVideo(video_path, time_txt_name, json_name, CONFIG.CAMERA_PARAMETER_PATH)


if __name__=="__main__":
    # videoname = sys.argv[1]
    # id = sys.argv[2]
    # dofile(videoname, id)
    list_file()