#-*- coding:utf-8 -*-
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger(__file__)

import os
import sys
import cv2
from math import *
import numpy as np
from PIL import Image

sys.path.append(os.getcwd() + '/ctpn')
from ctpn.text_detect import text_detect
from ctpn.text_detect import text_detect_without_img_drawed
from lib.fast_rcnn.config import cfg_from_file
from densenet.model import predict as keras_densenet

class OcrRunner:
    def __init__(self):
        # fast-rcnn 配置加载
        cfg_from_file('./ctpn/ctpn/text.yml')

    def sort_box(self, box):
        """ 
        对box进行排序
        """
        box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
        return box

    def dumpRotateImage(self, img, degree, pt1, pt2, pt3, pt4):
        """
        @img: 截取图片
        @img: 图片
        @degree: 倾斜角度
        @pt1: 左上角坐标
        @pt2: 右上角坐标
        @pt3: 左下角坐标
        @pt4: 右下角坐标
        """
        height, width = img.shape[:2]
        heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
        widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

        # 获得图像绕着中心点的旋转矩阵
        matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
        matRotation[0, 2] += (widthNew - width) // 2
        matRotation[1, 2] += (heightNew - height) // 2

        # 对图像进行仿射变化
        imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
        pt1 = list(pt1)
        pt3 = list(pt3)

        # numpy.dot 矩阵积
        [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
        [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))

        # shape 获取矩阵围堵
        # 获取矩阵前两维的维度，即图片的高和宽
        ydim, xdim = imgRotation.shape[:2]
        imgOut = imgRotation[max(1, int(pt1[1])) : min(ydim - 1, int(pt3[1])), max(1, int(pt1[0])) : min(xdim - 1, int(pt3[0]))]

        return imgOut

    def charRec(self, img, text_recs, adjust=False):
        """
        加载OCR模型，进行字符识别
        """
        results = {}
        # 图片的宽、高
        xDim, yDim = img.shape[1], img.shape[0]
            
        for index, rec in enumerate(text_recs):
            xlength = int((rec[6] - rec[0]) * 0.1)
            ylength = int((rec[7] - rec[1]) * 0.2)
            if adjust:
                pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
                pt2 = (rec[2], rec[3])
                pt3 = (min(rec[6] + xlength, xDim - 2), min(yDim - 2, rec[7] + ylength))
                pt4 = (rec[4], rec[5])
            else:
                pt1 = (max(1, rec[0]), max(1, rec[1]))
                pt2 = (rec[2], rec[3])
                pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
                pt4 = (rec[4], rec[5])
            
            # atan2() 返回给定的 X 及 Y 坐标值的反正切值
            # degrees() 将弧度转换为角度
            # 倾斜角度
            degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  # 图像倾斜角度

            # 截取图片
            partImg = dumpRotateImage(img, degree, pt1, pt2, pt3, pt4)

            # 过滤异常图片：高、宽 小于 1，或者 高 大于 宽
            if partImg.shape[0] < 1 or partImg.shape[1] < 1 or partImg.shape[0] > partImg.shape[1]:
                continue

            # 转换为 8 位像素的黑白色
            image = Image.fromarray(partImg).convert('L')

            # 识别文字
            text = keras_densenet(image)
            
            if len(text) > 0:
                results[index] = [rec]
                results[index].append(text)  
        
        return results

    def model(self, img, adjust=False):
        """
        @img: 图片
        @adjust: 是否调整文字识别结果
        """
        text_recs, img_framed, img = text_detect(img)
        text_recs = sort_box(text_recs)
        result = charRec(img, text_recs, adjust)
        return result, img_framed

    # 文字识别
    def modelWithoutImgDrawed(self, img, adjust=False):
        """
        @img: 图片
        @adjust: 是否调整文字识别结果
        """
        logger.debug(u">> 文字区域识别步骤 start")

        # 文字区域识别
        text_recs, img = text_detect_without_img_drawed(img)
        text_recs = sort_box(text_recs)

        logger.debug(u">> 文字区域识别步骤 over")

        logger.debug(u">> 字符识别步骤 start")

        # 字符识别
        result = charRec(img, text_recs, adjust)

        logger.debug(u">> 字符识别步骤 over")

        return result

