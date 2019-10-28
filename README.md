# 简介
基于Tensorflow和Keras实现端到端的不定长中文字符检测和识别。

该项目基于 `https://github.com/chineseocr/chinese-ocr` 改进和优化。我们主要做了一些以下工作：
+ 代码注释
+ 依赖升级
+ 文档优化

* 文本检测：CTPN
* 文本识别：DenseNet + CTC

依赖的算法和技术：
+ CTPN
+ Tensorflow
+ keras
+ fast-rcnn

## 目录结构
```txt
/

/ctpn                           # 场景文本检测
/ctpn/checkpoints               # 已经训练好的模型放置目录
/ctpn/ctpn                      # 
/ctpn/data                      # 
/ctpn/lib                       # 第三方依赖
/ctpn/prepare_training_data     # 
/ctpn/text_detect.py            # 场景文本检测脚本

/densenet                       # 字符识别

/img                            # 示例图片

/test
/test/images                    # 测试图片数据
/test/result                    # 测试结果输出目录
/train                          # 训练脚本及数据目录

/demo.py                        # 测试demo
/ocr.py                         # 文字识别

/setup-cpu.sh                   # cpu 方式设置脚本
/setup-gpu.sh                   # gpu 方式设置脚本
```


# 使用 conda 部署（推荐）

> 建议在 linux 下开发

1. 安装 [Anaconda](https://www.anaconda.com/)。（强烈建议）
    > + Anaconda是一个方便的 python 包管理和环境管理软件，一般用来配置不同的项目环境。使用 Anaconda 可以避免不同工程的依赖冲突。
    > + 使用 Python 2.7 版本的 Anaconda
    > + 下载地址（清华源）：https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/
2. 使用清华的 Anaconda 源（可选）
```sh
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```
3. 运行配置脚本 `setup.sh`

``` Bash
sh setup.sh
```
> 注：CPU环境执行前需注释掉for gpu部分，并解开for cpu部分的注释

# Demo
将测试图片放入 `test/image` 目录，检测结果会保存到 `test/output` 中

``` Bash
python demo.py
```

> 必须指定： `CUDA_HOME` 环境变量

# 模型训练

### CTPN训练
详见ctpn/README.md

### DenseNet + CTC训练

#### 1. 数据准备

数据集：https://pan.baidu.com/s/1QkI7kjah8SPHwOQ40rS1Pw (密码：lu7m)
* 共约364万张图片，按照99:1划分成训练集和验证集
* 数据利用中文语料库（新闻 + 文言文），通过字体、大小、灰度、模糊、透视、拉伸等变化随机生成
* 包含汉字、英文字母、数字和标点共5990个字符
* 每个样本固定10个字符，字符随机截取自语料库中的句子
* 图片分辨率统一为280x32

图片解压后放置到train/images目录下，描述文件放到train目录下

#### 2. 训练

``` Bash
cd train
python train.py
```

#### 3. 结果

| val acc | predict | model |
| -----------| ---------- | -----------|
| 0.983 | 8ms | 18.9MB |

* GPU: GTX TITAN X
* Keras Backend: Tensorflow

#### 4. 生成自己的样本

可参考[SynthText_Chinese_version](https://github.com/JarveeLee/SynthText_Chinese_version)，[TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator)和[text_renderer](https://github.com/Sanster/text_renderer)

# 效果展示
<div>
<img width="420" height="420" src="https://gitlab.91chengguo.com/Orange-library/chinese-ocr/raw/master/img/demo_detect.jpg"/>
<img width="420" height="420" src="https://gitlab.91chengguo.com/Orange-library/chinese-ocr/raw/master/img/demo_rec.jpg"/>
</div>

# 鸣谢
该项目基于 `https://github.com/chineseocr/chinese-ocr` 项目改动，十分感谢源作者的付出。

> 注：
> 由于原仓库已被作者删除，我们只能发布当时下载的源码，不能发布原始完整的 git 记录，请谅解。如有侵权，请及时联系我们。

# 参考

+ [1] https://github.com/eragonruan/text-detection-ctpn
+ [2] https://github.com/senlinuc/caffe_ocr
+ [3] https://github.com/chineseocr/chinese-ocr
+ [4] https://github.com/xiaomaxiao/keras_ocr
