#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.io import cache_url
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis_new as vis_utils

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def drawBoxOnImg(img,x,y,w,h,p_x,p_y,num):
    cv2.rectangle(img,(int(x),int(y)),(int(x+w),int(y+h)),(127,255,0),5)
    cv2.circle(img, (int(p_x),int(p_y)), 5, (255,0,0),-1) 
    img_path = '/data/pic/box_'+str(num)+'.jpg'
    cv2.imwrite(img_path,img, [int( cv2.IMWRITE_JPEG_QUALITY), 95])

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)

    args.cfg = '/detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml'
    args.weights = '/data/workspace/fbdet/models/mask_rcnn_R_101_FPN_2x/model_final.pkl'

    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]
    im_list = []
    im_list.append('/data/workspace/fbdet/test_pic/11.jpg')
    video = cv2.VideoCapture('/data/pic/valid_video_00.avi')
    frame = 0
    while(True):
        if frame > 0:
            break
        # ret, im = video.read()
        # if im is None or ret is None:
        #     print("video.read() fail || video.read() is end!")
        #     break
        im = cv2.imread(im_list[0])
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        
        print('-----------------------------',frame)
        boxs_list = vis_utils.get_boxes_image(cls_boxes,cls_segms,cls_keyps,thresh=0.7,dataset=dummy_coco_dataset)
        print(boxs_list)
        print('-----------------------------')
        for i in range(len(boxs_list)):
            box = boxs_list[i]
            drawBoxOnImg(im,box[1],box[2],box[3],box[4],0,0,frame)
        frame+=1

workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
utils.logging.setup_logging(__name__)
logger = logging.getLogger(__name__)
#args = parse_args()
# args.cfg = '/detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml'
# args.weights = '/data/workspace/fbdet/models/mask_rcnn_R_101_FPN_2x/model_final.pkl'
cfg_path = '/detectron/configs/12_2017_baselines/e2e_faster_rcnn_X-101-64x4d-FPN_1x.yaml'
weights_path = '/data/workspace/fbdet/models/X_101_64x4d_FPN_faster/model_final.pkl'
merge_cfg_from_file(cfg_path)
cfg.NUM_GPUS = 1
weights1 = cache_url(weights_path, cfg.DOWNLOAD_CACHE)
assert_and_infer_cfg(cache_urls=False)
model = infer_engine.initialize_model_from_cfg(weights1)
dummy_coco_dataset = dummy_datasets.get_coco_dataset()


def pipeline_mask(im):

    if im is None:
        print("im is null!")

    timers = defaultdict(Timer)
    #t = time.time()
    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
            model, im, None, timers=timers
        )
    #logger.info('Inference time: {:.3f}s'.format(time.time() - t))
    #for k, v in timers.items():
    #    logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

    #print('===========================boxs_list:')
    boxs_list = vis_utils.get_boxes_image(cls_boxes,cls_segms,cls_keyps,thresh=0.7,dataset=dummy_coco_dataset)
    #print(boxs_list)

    #for i in range(len(boxs_list)):
    #    box = boxs_list[i]
        #drawBoxOnImg(im,box[1],box[2],box[3],box[4],0,0,frame)
    #frame+=1
    return boxs_list

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    im_list = []
    im_list.append('/data/workspace/fbdet/test_pic/11.jpg')
    im = cv2.imread(im_list[0])
    pipeline_mask(im)
    #main(args)
