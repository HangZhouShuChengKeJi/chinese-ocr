#!/bin/sh

# # 创建 chinese-ocr 环境（创建一次即可）
conda create -n chinese-ocr-cpu python=3.5

# # 启用 chinese-ocr 环境
conda activate chinese-ocr-cpu

# 升级 pip 版本
python -m pip install -i https://mirrors.huaweicloud.com/repository/pypi/simple pip --upgrade

# 使用华为的 pip 源
pip config set global.index-url https://mirrors.huaweicloud.com/repository/pypi/simple

# 安装依赖

pip install numpy==1.15.4
pip install scipy matplotlib pillow
pip install easydict opencv-python keras h5py PyYAML
pip install cython==0.24

pip install tensorflow==1.3.0
cd ./ctpn/lib/utils/ && sh ./make_cpu.sh
