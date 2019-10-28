#-*- coding:utf-8 -*-
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger(__file__)

import os
from ocr import OcrRunner
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob


# 输入图片文件集合
input_img_files = glob('./test/image/*.*')

# 输出结果目录
output_dir = './test/output'

if __name__ == '__main__':

    logger.debug(u">>>> 开始执行")

    logger.debug(u">>>> 初始化...")

    # 清空输出结果目录
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    ocrRunner = OcrRunner()

    logger.debug(u">>>> 初始化完成")

    for image_file in sorted(input_img_files):
        logger.debug(u"\t>> 处理图片：{}".format(image_file))
        # 加载图片
        image = np.array(Image.open(image_file).convert('RGB'))

        # 文件名
        image_name = image_file.split('/')[-1]

        # 输出文件
        output_file = os.path.join(output_dir, image_name)

        # 开始计时
        startTime = time.time()

        # 进行 ocr 识别
        # result, image_framed = ocr.model(image)
        result = ocrRunner.modelWithoutImgDrawed(image)

        # 结束计时
        endTime = time.time()

        logger.debug(u"OCR 完成，耗时： {:.3f}s ".format(endTime - startTime))

        logger.debug(u"OCR 识别结果: ")

        for key in result:
            position = result[key][0]
            ocrTxt = result[key][1]
            logger.debug(u"\t>> 文字区域坐标：[({}, {}), ({}, {}), ({}, {}), ({}, {})]".format(position[0], position[1], position[2], position[3], position[4], position[5], position[6], position[7]))
            logger.debug(u"\t>> 识别文字内容：{}".format(ocrTxt))
            # logger.debug("\t>> %s" % (ocrTxt))
            
        # 输出框选后的图片文件
        # Image.fromarray(image_framed).save(output_file)

