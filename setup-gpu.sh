#!/bin/sh

# 创建 chinese-ocr 环境（创建一次即可）
conda create -n chinese-ocr-gpu python=2.7

# 启用 chinese-ocr 环境
conda activate chinese-ocr-gpu

# 升级 pip 版本
python -m pip install -i https://mirrors.huaweicloud.com/repository/pypi/simple pip --upgrade

# 使用华为的 pip 源
pip config set global.index-url https://mirrors.huaweicloud.com/repository/pypi/simple

# 安装依赖

pip install numpy==1.15.4
pip install scipy matplotlib pillow
pip install easydict opencv-python keras h5py PyYAML
pip install cython==0.24

############ for gpu ############

# 依赖 cuda 8.0
conda install -n chinese-ocr cudatoolkit=8.0
# 依赖 cudnn 6.0
conda install -n chinese-ocr cudnn=6.0

pip install tensorflow-gpu==1.3.0
chmod +x ./ctpn/lib/utils/make.sh
cd ./ctpn/lib/utils/ && sh ./make.sh

