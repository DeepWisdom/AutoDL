[English](./README_EN.md) | 简体中文

<p align="center">

[![GitHub issues](https://img.shields.io/github/issues/DeepWisdom/AutoDL)](https://github.com/DeepWisdom/AutoDL/issues)
[![GitHub forks](https://img.shields.io/github/forks/DeepWisdom/AutoDL)](https://github.com/DeepWisdom/AutoDL/network)
[![GitHub stars](https://img.shields.io/github/stars/DeepWisdom/AutoDL)](https://github.com/DeepWisdom/AutoDL/stargazers)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/deepwisdom/AutoDL)
[![GitHub license](https://img.shields.io/github/license/DeepWisdom/AutoDL)](https://github.com/DeepWisdom/AutoDL/blob/master/LICENSE)
![img](https://img.shields.io/badge/python-3.5-brightgreen)
[![img](https://img.shields.io/badge/chat-wechat-green)](https://github.com/DeepWisdom/AutoDL#%E5%8A%A0%E5%85%A5%E7%A4%BE%E5%8C%BA)
</p>



# 1. NeurIPS AutoDL Challenge 冠军方案

![img](https://raw.githubusercontent.com/DeepWisdom/AutoDL/pip/assets/autodl_logo_full.png)

[AutoDL Challenge@NeurIPS](https://autodl.chalearn.org/neurips2019) 冠军方案，竞赛细节参见 [AutoDL Competition](https://autodl.lri.fr/competitions/162)。

## 1.1. AutoDL是什么？

AutoDL聚焦于自动进行任意模态（图像、视频、语音、文本、表格数据）多标签分类的通用算法，可以用一套标准算法流解决现实世界的复杂分类问题，解决调数据、特征、模型、超参等烦恼，最短10秒就可以做出性能优异的分类器。本工程在**不同领域的24个离线数据集、15个线上数据集都获得了极为优异的成绩**。AutoDL拥有以下特性：

☕ **全自动**：全自动深度学习/机器学习框架，全流程无需人工干预。数据、特征、模型的所有细节都已调节至最佳，统一解决了资源受限、数据倾斜、小数据、特征工程、模型选型、网络结构优化、超参搜索等问题。**只需要准备数据，开始AutoDL，然后喝一杯咖啡**。

🌌 **通用性**：支持**任意**模态，包括图像、视频、音频、文本和结构化表格数据，支持**任意多标签分类问题**，包括二分类、多分类、多标签分类。它在**不同领域**都获得了极其优异的成绩，如行人识别、行人动作识别、人脸识别、声纹识别、音乐分类、口音分类、语言分类、情感分类、邮件分类、新闻分类、广告优化、推荐系统、搜索引擎、精准营销等等。

👍 **效果出色**：AutoDL竞赛获得压倒性优势的冠军方案，包含对传统机器学习模型和最新深度学习模型支持。模型库包括从LR/SVM/LGB/CGB/XGB到ResNet*/MC3/DNN/ThinResnet*/TextCNN/RCNN/GRU/BERT等优选出的冠军模型。

⚡ **极速/实时**：最快只需十秒即可获得极具竞争力的模型性能。结果实时刷新（秒级），无需等待即可获得模型实时效果反馈。

## 1.2. 目录
<!-- TOC -->

- [1. NeurIPS AutoDL Challenge 冠军方案](#1-neurips-autodl-challenge-%e5%86%a0%e5%86%9b%e6%96%b9%e6%a1%88)
  - [1.1. AutoDL是什么？](#11-autodl%e6%98%af%e4%bb%80%e4%b9%88)
  - [1.2. 目录](#12-%e7%9b%ae%e5%bd%95)
  - [1.3. 效果](#13-%e6%95%88%e6%9e%9c)
  - [1.4. AutoDL竞赛代码使用说明](#14-autodl%e7%ab%9e%e8%b5%9b%e4%bb%a3%e7%a0%81%e4%bd%bf%e7%94%a8%e8%af%b4%e6%98%8e)
    - [1.4.1. 使用效果示例（横轴为对数时间轴，纵轴为AUC）](#141-%e4%bd%bf%e7%94%a8%e6%95%88%e6%9e%9c%e7%a4%ba%e4%be%8b%e6%a8%aa%e8%bd%b4%e4%b8%ba%e5%af%b9%e6%95%b0%e6%97%b6%e9%97%b4%e8%bd%b4%e7%ba%b5%e8%bd%b4%e4%b8%baauc)
  - [1.5. 安装](#15-%e5%ae%89%e8%a3%85)
    - [1.5.1. pip 安装](#151-pip-%e5%ae%89%e8%a3%85)
  - [1.6. 快速上手](#16-%e5%bf%ab%e9%80%9f%e4%b8%8a%e6%89%8b)
    - [1.6.1. 快速上手之AutoDL本地效果测试](#161-%e5%bf%ab%e9%80%9f%e4%b8%8a%e6%89%8b%e4%b9%8bautodl%e6%9c%ac%e5%9c%b0%e6%95%88%e6%9e%9c%e6%b5%8b%e8%af%95)
    - [1.6.2. 快速上手之图像分类](#162-%e5%bf%ab%e9%80%9f%e4%b8%8a%e6%89%8b%e4%b9%8b%e5%9b%be%e5%83%8f%e5%88%86%e7%b1%bb)
    - [1.6.3. 快速上手之视频分类](#163-%e5%bf%ab%e9%80%9f%e4%b8%8a%e6%89%8b%e4%b9%8b%e8%a7%86%e9%a2%91%e5%88%86%e7%b1%bb)
    - [1.6.4. 快速上手之音频分类](#164-%e5%bf%ab%e9%80%9f%e4%b8%8a%e6%89%8b%e4%b9%8b%e9%9f%b3%e9%a2%91%e5%88%86%e7%b1%bb)
    - [1.6.5. 快速上手之文本分类](#165-%e5%bf%ab%e9%80%9f%e4%b8%8a%e6%89%8b%e4%b9%8b%e6%96%87%e6%9c%ac%e5%88%86%e7%b1%bb)
    - [1.6.6. 快速上手之表格分类](#166-%e5%bf%ab%e9%80%9f%e4%b8%8a%e6%89%8b%e4%b9%8b%e8%a1%a8%e6%a0%bc%e5%88%86%e7%b1%bb)
  - [1.7. 可用数据集](#17-%e5%8f%af%e7%94%a8%e6%95%b0%e6%8d%ae%e9%9b%86)
    - [1.7.1. (可选) 下载数据集](#171-%e5%8f%af%e9%80%89-%e4%b8%8b%e8%bd%bd%e6%95%b0%e6%8d%ae%e9%9b%86)
    - [1.7.2. 公共数据集信息](#172-%e5%85%ac%e5%85%b1%e6%95%b0%e6%8d%ae%e9%9b%86%e4%bf%a1%e6%81%af)
  - [1.8. 贡献代码](#18-%e8%b4%a1%e7%8c%ae%e4%bb%a3%e7%a0%81)
  - [1.9. 加入社区](#19-%e5%8a%a0%e5%85%a5%e7%a4%be%e5%8c%ba)
  - [1.10. 开源协议](#110-%e5%bc%80%e6%ba%90%e5%8d%8f%e8%ae%ae)

<!-- /TOC -->


## 1.3. 效果
- **预赛榜单（DeepWisdom总分第一，平均排名1.2，在5个数据集中取得了4项第一）**
![img](https://raw.githubusercontent.com/DeepWisdom/AutoDL/pip/assets/feedback-lb.png)

- **决赛榜单（DeepWisdom总分第一，平均排名1.8，在10个数据集中取得了7项第一）**
![img](https://raw.githubusercontent.com/DeepWisdom/AutoDL/pip/assets/final-lb-visual.png)


## 1.4. AutoDL竞赛代码使用说明

1. 基础环境
    ```shell script
    python>=3.5
    CUDA 10
    cuDNN 7.5
    ```

2. clone仓库 
    ```
    cd <path_to_your_directory>
    git clone https://github.com/DeepWisdom/AutoDL.git
    ```
3. 预训练模型准备
下载模型 [speech_model.h5](https://github.com/DeepWisdom/AutoDL/releases/download/opensource/thin_resnet34.h5) 放至 `AutoDL_sample_code_submission/at_speech/pretrained_models/` 目录。

4. 可选：使用与竞赛同步的docker环境 
    - CPU
    ```
    cd path/to/autodl/
    docker run -it -v "$(pwd):/app/codalab" -p 8888:8888 evariste/autodl:cpu-latest
    ```
    - GPU
    ```
    nvidia-docker run -it -v "$(pwd):/app/codalab" -p 8888:8888 evariste/autodl:gpu-latest
    ```
5. 数据集准备：使用 `AutoDL_sample_data` 中样例数据集，或批量下载竞赛公开数据集。

6. 进行本地测试
    ```
    python run_local_test.py
    ```
本地测试完整使用。
    ```
    python run_local_test.py -dataset_dir='AutoDL_sample_data/miniciao' -code_dir='AutoDL_sample_code_submission'
    ```
您可在 `AutoDL_scoring_output/` 目录中查看实时学习曲线反馈的HTML页面。

细节可参考 [AutoDL Challenge official starting_kit](https://github.com/zhengying-liu/autodl_starting_kit_stable).

### 1.4.1. 使用效果示例（横轴为对数时间轴，纵轴为AUC）

![img](https://github.com/DeepWisdom/AutoDL/tree/pip/assets/AutoDL-performance-example.png)

可以看出，在五个不同模态的数据集下，AutoDL算法流都获得了极为出色的全时期效果，可以在极短的时间内达到极高的精度。

## 1.5. 安装 

本仓库在 Python 3.6+, PyTorch 1.3.1 和 TensorFlow 1.15上测试.

你应该在[虚拟环境](https://docs.python.org/3/library/venv.html) 中安装autodl。
如果对虚拟环境不熟悉，请看 [用户指导](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

用合适的Python版本创建虚拟环境，然后激活它。

### 1.5.1. pip 安装

AutoDL 能用以下方式安装:

```bash
pip install autodl-gpu
pip install autodl-gpu==1.0.0
```

## 1.6. 快速上手
### 1.6.1. 快速上手之AutoDL本地效果测试
参见 [快速上手之AutoDL本地效果测试](https://github.com/DeepWisdom/AutoDL/tree/pip/docs/run_local_test_tutorial_chn.md)

### 1.6.2. 快速上手之图像分类
参见 [快速上手之图像分类](https://github.com/DeepWisdom/AutoDL/tree/pip/docs/image_classification_tutorial_chn.md)

### 1.6.3. 快速上手之视频分类
参见 [快速上手之视频分类](https://github.com/DeepWisdom/AutoDL/tree/pip/docs/video_classification_tutorial_chn.md)

### 1.6.4. 快速上手之音频分类
参见 [快速上手之音频分类](https://github.com/DeepWisdom/AutoDL/tree/pip/docs/speech_classification_tutorial_chn.md)

### 1.6.5. 快速上手之文本分类
参见 [快速上手之文本分类](https://github.com/DeepWisdom/AutoDL/tree/pip/docs/text_classification_tutorial_chn.md)

### 1.6.6. 快速上手之表格分类
参见 [快速上手之表格分类](https://github.com/DeepWisdom/AutoDL/tree/pip/docs/tabular_classification_tutorial_chn.md)





## 1.7. 可用数据集
### 1.7.1. (可选) 下载数据集
    ```bash
    python download_public_datasets.py
    ```

### 1.7.2. 公共数据集信息
| #   | Name     | Type    | Domain   | Size   | Source      | Data (w/o test labels) | Test labels       |
| --- | -------- | ------- | -------- | ------ | ----------- | ---------------------- | ----------------- |
| 1   | Munster  | Image   | HWR      | 18 MB  | MNIST       | munster.data           | munster.solution  |
| 2   | City     | Image   | Objects  | 128 MB | Cifar-10    | city.data              | city.solution     |
| 3   | Chucky   | Image   | Objects  | 128 MB | Cifar-100   | chucky.data            | chucky.solution   |
| 4   | Pedro    | Image   | People   | 377 MB | PA-100K     | pedro.data             | pedro.solution    |
| 5   | Decal    | Image   | Aerial   | 73 MB  | NWPU VHR-10 | decal.data             | decal.solution    |
| 6   | Hammer   | Image   | Medical  | 111 MB | Ham10000    | hammer.data            | hammer.solution   |
| 7   | Kreatur  | Video   | Action   | 469 MB | KTH         | kreatur.data           | kreatur.solution  |
| 8   | Kreatur3 | Video   | Action   | 588 MB | KTH         | kreatur3.data          | kreatur3.solution |
| 9   | Kraut    | Video   | Action   | 1.9 GB | KTH         | kraut.data             | kraut.solution    |
| 10  | Katze    | Video   | Action   | 1.9 GB | KTH         | katze.data             | katze.solution    |
| 11  | data01   | Speech  | Speaker  | 1.8 GB | --          | data01.data            | data01.solution   |
| 12  | data02   | Speech  | Emotion  | 53 MB  | --          | data02.data            | data02.solution   |
| 13  | data03   | Speech  | Accent   | 1.8 GB | --          | data03.data            | data03.solution   |
| 14  | data04   | Speech  | Genre    | 469 MB | --          | data04.data            | data04.solution   |
| 15  | data05   | Speech  | Language | 208 MB | --          | data05.data            | data05.solution   |
| 16  | O1       | Text    | Comments | 828 KB | --          | O1.data                | O1.solution       |
| 17  | O2       | Text    | Emotion  | 25 MB  | --          | O2.data                | O2.solution       |
| 18  | O3       | Text    | News     | 88 MB  | --          | O3.data                | O3.solution       |
| 19  | O4       | Text    | Spam     | 87 MB  | --          | O4.data                | O4.solution       |
| 20  | O5       | Text    | News     | 14 MB  | --          | O5.data                | O5.solution       |
| 21  | Adult    | Tabular | Census   | 2 MB   | Adult       | adult.data             | adult.solution    |
| 22  | Dilbert  | Tabular | --       | 162 MB | --          | dilbert.data           | dilbert.solution  |
| 23  | Digits   | Tabular | HWR      | 137 MB | MNIST       | digits.data            | digits.solution   |
| 24  | Madeline | Tabular | --       | 2.6 MB | --          | madeline.data          | madeline.solution |


## 1.8. 贡献代码 

❤️ 请毫不犹豫参加贡献 [Open an issue](https://github.com/DeepWisdom/AutoDL/issues/new) 或提交 PRs。

## 1.9. 加入社区
<img src="https://raw.githubusercontent.com/DeepWisdom/AutoDL/pip/assets/WechatIMG15.png" width = "500" height = "200" alt="AutoDL Community" align=center />

## 1.10. 开源协议 
[Apache License 2.0](https://github.com/DeepWisdom/AutoDL/blob/master/LICENSE)
