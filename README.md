[English](./README_EN.md) | 简体中文

# NeurIPS AutoDL Challenge 冠军方案

![img](assets/autodl_logo_full.png)

[AutoDL Challenge@NeurIPS](https://autodl.chalearn.org/neurips2019) 冠军方案，竞赛细节参见 [AutoDL Competition](https://autodl.lri.fr/competitions/162)。

## AutoDL是什么？

现实世界有一系列常见且棘手的问题，如资源（CPU/内存）受限、样本不平衡、特征需定制、模型需选型、网络结构细节需调优、预训练领域敏感、超参数敏感等等，这些问题如何高效解决？

AutoDL聚焦于自动进行任意模态（图像、视频、语音、文本、表格数据）多标签分类的通用算法，可以用一套标准算法流解决现实世界的复杂分类问题，最短10秒就可以做出性能优异的分类器。本工程在不同领域的24个离线数据集、15个线上数据集都获得了极为优异的成绩，击败了众多世界顶级选手。


## 目录
<!-- TOC -->

- [NeurIPS AutoDL Challenge 冠军方案](#neurips-autodl-challenge-%e5%86%a0%e5%86%9b%e6%96%b9%e6%a1%88)
  - [AutoDL是什么？](AutoDL是什么？)
  - [目录](#%e7%9b%ae%e5%bd%95)
  - [效果展示](#%e6%95%88%e6%9e%9c%e5%b1%95%e7%a4%ba)
  - [特性](#%e7%89%b9%e6%80%a7)
  - [公共数据集](#%e5%85%ac%e5%85%b1%e6%95%b0%e6%8d%ae%e9%9b%86)
    - [(可选) 下载数据集](#%e5%8f%af%e9%80%89-%e4%b8%8b%e8%bd%bd%e6%95%b0%e6%8d%ae%e9%9b%86)
    - [公共数据集信息](#%e5%85%ac%e5%85%b1%e6%95%b0%e6%8d%ae%e9%9b%86%e4%bf%a1%e6%81%af)
  - [本地开发测试说明](#%e6%9c%ac%e5%9c%b0%e5%bc%80%e5%8f%91%e6%b5%8b%e8%af%95%e8%af%b4%e6%98%8e)
  - [贡献代码](#%e8%b4%a1%e7%8c%ae%e4%bb%a3%e7%a0%81)
  - [联系我们](#%e8%81%94%e7%b3%bb%e6%88%91%e4%bb%ac)
  - [加入社区](#%e5%8a%a0%e5%85%a5%e7%a4%be%e5%8c%ba)
  - [开源协议](#%e5%bc%80%e6%ba%90%e5%8d%8f%e8%ae%ae)

<!-- /TOC -->


## 效果展示
- **预赛榜单（DeepWisdom总分第一，平均排名1.2，在5个数据集中取得了4项第一）**
![img](assets/feedback-lb.png)

- **决赛榜单（DeepWisdom总分第一，平均排名1.8，在10个数据集中取得了7项第一）**
![img](assets/final-lb-visual.png)

## 特性 
- ☕ **全自动**: 全自动深度学习/机器学习框架，全流程无需人工干预。
- 🌏 **通用性**: 支持**任意**模态，包括图像、视频、音频、文本和结构化表格数据，支持**任意多标签分类问题**。
- 👍 **效果出色**: AutoDL竞赛获得压倒性优势的冠军方案，包含对传统机器学习模型和最新深度学习模型支持。
- 🔑 **开箱即用**: 只需准备数据，然后点 *Run*!
- ⚡ **极速**: 最快只需十秒即可获得极具竞争力的模型性能。
- 🚀 **实时反馈**: 结果实时刷新（秒级），无需等待即可获得模型实时效果反馈。


## 公共数据集
### (可选) 下载数据集
```bash
python download_public_datasets.py
```

### 公共数据集信息
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


## 本地开发测试说明
1. clone仓库 
```
cd <path_to_your_directory>
git clone https://github.com/DeepWisdom/AutoDL.git
```
2. 预训练模型准备
下载模型 [speech_model.h5](https://github.com/DeepWisdom/AutoDL/releases/download/opensource/thin_resnet34.h5) 放至 `AutoDL_sample_code_submission/at_speech/pretrained_models/` 目录。

3. 可选：使用与竞赛同步的docker环境 
    - CPU
    ```
    cd path/to/autodl/
    docker run -it -v "$(pwd):/app/codalab" -p 8888:8888 evariste/autodl:cpu-latest
    ```
    - GPU
    ```
    nvidia-docker run -it -v "$(pwd):/app/codalab" -p 8888:8888 evariste/autodl:gpu-latest
    ```
4. 数据集准备：使用 `AutoDL_sample_data` 中样例数据集，或批量下载竞赛公开数据集。

5. 进行本地测试
```
python run_local_test.py
```
本地测试完整使用。
```
python run_local_test.py -dataset_dir='AutoDL_sample_data/miniciao' -code_dir='AutoDL_sample_code_submission'
```
您可在 `AutoDL_scoring_output/` 目录中查看实时学习曲线反馈的HTML页面。

细节可参考 [AutoDL Challenge official starting_kit](https://github.com/zhengying-liu/autodl_starting_kit_stable).


## 贡献代码 

请毫不犹豫参加贡献 [Open an issue](https://github.com/DeepWisdom/AutoDL/issues/new) 或提交 PRs。

## 联系我们 

[![img](https://github.com/DeepWisdom/AutoDL/blob/master/assets/deepwisdom-logo-white.svg "title")](http://fuzhi.ai/)

## 加入社区

<img src="./assets/WechatIMG15.png" width = "500" height = "200" alt="AutoDL社区" align=center />

## 开源协议 
[Apache License 2.0](https://github.com/DeepWisdom/AutoDL/blob/master/LICENSE)
