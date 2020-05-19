English | [简体中文](./README.md)

<p align="center">

[![GitHub issues](https://img.shields.io/github/issues/DeepWisdom/AutoDL)](https://github.com/DeepWisdom/AutoDL/issues)
[![GitHub forks](https://img.shields.io/github/forks/DeepWisdom/AutoDL)](https://github.com/DeepWisdom/AutoDL/network)
[![GitHub stars](https://img.shields.io/github/stars/DeepWisdom/AutoDL)](https://github.com/DeepWisdom/AutoDL/stargazers)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/deepwisdom/AutoDL)
[![GitHub license](https://img.shields.io/github/license/DeepWisdom/AutoDL)](https://github.com/DeepWisdom/AutoDL/blob/master/LICENSE)
![img](https://img.shields.io/badge/python-3.5-brightgreen)
[![img](https://img.shields.io/badge/chat-wechat-green)](https://github.com/DeepWisdom/AutoDL#%E5%8A%A0%E5%85%A5%E7%A4%BE%E5%8C%BA)
</p>

<!-- # 1. NeurIPS AutoDL Challenge 1'st Solution -->

![img](https://github.com/DeepWisdom/AutoDL/tree/pip/assets/autodl_logo_full.png)
![img](https://raw.githubusercontent.com/DeepWisdom/AutoDL/pip/assets/autodl_logo_full.png)


**1st** solution for [AutoDL Challenge@NeurIPS](https://autodl.chalearn.org/neurips2019), competition rules can be found at [AutoDL Competition](https://autodl.lri.fr/competitions/162).

# 1. Motivation 
There exists a series of common and tough problems in the real world, such as limited resources (CPU/ memory), skewed data, hand-craft features, model selection, network architecture details tuning, sensitivity of pre-trained models, sensitivity of hyperparameters and so on. How to solve them wholly and efficiently?


# 2. Solution 
AutoDL concentrates on developing generic algorithms for multi-label classification problems in ANY modalities: image, video, speech, text and tabular data without ANY human intervention.  **Ten seconds** at the soonest, our solution achieved SOTA performances on all the 24 offline datasets and 15 online datasets, beating a number of top players in the world.


# 3. Table of Contents
<!-- TOC -->

- [1. Motivation](#1-motivation)
- [2. Solution](#2-solution)
- [3. Table of Contents](#3-table-of-contents)
- [4. Features](#4-features)
- [5. Evaluation](#5-evaluation)
- [6. Installation](#6-installation)
  - [6.1. With pip](#61-with-pip)
- [7. Quick Tour](#7-quick-tour)
  - [7.1. Run local test tour](#71-run-local-test-tour)
  - [7.2. Tour of Image Classification](#72-tour-of-image-classification)
  - [7.3. Tour of Video Classification](#73-tour-of-video-classification)
  - [7.4. Tour of Speech Classification](#74-tour-of-speech-classification)
  - [7.5. Tour of Text Classification](#75-tour-of-text-classification)
  - [7.6. Tour of Tabular Classification](#76-tour-of-tabular-classification)
- [8. Public Datasets](#8-public-datasets)
  - [8.1. Optional: Download public datasets](#81-optional-download-public-datasets)
  - [8.2. Public datasets sample info](#82-public-datasets-sample-info)
- [9. Usage for AutoDL local development and testing](#9-usage-for-autodl-local-development-and-testing)
- [10. Contributing](#10-contributing)
- [11. Contact us](#11-contact-us)
- [12. Join the Community](#12-join-the-community)
- [13. License](#13-license)

<!-- /TOC -->



# 4. Features
- **Full-AutoML/AutoDL**: Fully automated Deep Learning without ANY human intervention covering the whole pipelines.
- **Generic & Universal**: Supporting ANY modality(image, video, speech, text, tabular) data, and **ANY** classification problems including binary-class, multi-class and multi-label problems.
- **SOTA**: Winner solution of AutoDL challenge, involving both tranditional machine learning models and deep learning model backbones. 
- **Out-of-the-Box**: You can use the solution out-of-the-box.
- **Fast**: You can train your model in **ten seconds** at the soonest to get highly competitive performance.
- **Real-time**: You can get the performance feedback(AUC score) in real time.



# 5. Evaluation


- **Feedback-phase leaderboard: DeepWisdom Top 1, average rank 1.2, won 4 out of 5 datasets.**
![img](https://raw.githubusercontent.com/DeepWisdom/AutoDL/pip/assets/feedback-lb.png)

- **Final-phase leaderboard visualization: DeepWisdom Top 1, average rank 1.2, won 7 out of 10 datasets.**
![img](https://raw.githubusercontent.com/DeepWisdom/AutoDL/pip/assets/final-lb-visual.png)


# 6. Installation

This repo is tested on Python 3.6+, PyTorch 1.0.0+ and TensorFlow 2.0.

You should install AutoDL in a [virtual environment](https://docs.python.org/3/library/venv.html). If you're unfamiliar with Python virtual environments, check out the [user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

Create a virtual environment with the version of Python you're going to use and activate it.

Now, if you want to use AutoDL, you can install it with pip.

## 6.1. With pip

AutoDL can be installed using pip as follows:

```bash
pip install autodl-gpu 
```

# 7. Quick Tour
## 7.1. Run local test tour
see [Quick Tour - Run local test tour](https://github.com/DeepWisdom/AutoDL/tree/pip/docs/run_local_test_tutorial_chn.md).


## 7.2. Tour of Image Classification
see [Quick Tour - Image Classification Demo](https://github.com/DeepWisdom/AutoDL/tree/pip/docs/image_classification_tutorial_chn.md).

## 7.3. Tour of Video Classification
see [Quick Tour - Video Classification Demo](https://github.com/DeepWisdom/AutoDL/tree/pip/docs/video_classification_tutorial_chn.md).

## 7.4. Tour of Speech Classification
see [Quick Tour - Speech Classification Demo](https://github.com/DeepWisdom/AutoDL/tree/pip/docs/speech_classification_tutorial_chn.md).

## 7.5. Tour of Text Classification
see [Quick Tour - Text Classification Demo](https://github.com/DeepWisdom/AutoDL/tree/pip/docs/text_classification_tutorial_chn.md).

## 7.6. Tour of Tabular Classification
see [Quick Tour - Tabular Classification Demo](https://github.com/DeepWisdom/AutoDL/tree/pip/docs/tabular_classification_tutorial_chn.md).


# 8. Public Datasets
## 8.1. Optional: Download public datasets
```bash
python download_public_datasets.py
```

## 8.2. Public datasets sample info 
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
| 12  | data02   | Speech  | Emotion  | 53 MB  | --          | data02.data            | dat02.solution    |
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



# 9. Usage for AutoDL local development and testing
1. Git clone the repo
```
cd <path_to_your_directory>
git clone https://github.com/DeepWisdom/AutoDL.git
```
2. Prepare pretrained models.
Download model [speech_model.h5](https://github.com/DeepWisdom/AutoDL/releases/download/opensource/thin_resnet34.h5) and put it to `AutoDL_sample_code_submission/at_speech/pretrained_models/` directory.

3. Optional: run in the exact same environment as on the challenge platform with docker. 
    - CPU
    ```
    cd path/to/autodl/
    docker run -it -v "$(pwd):/app/codalab" -p 8888:8888 evariste/autodl:cpu-latest
    ```
    - GPU
    ```
    nvidia-docker run -it -v "$(pwd):/app/codalab" -p 8888:8888 evariste/autodl:gpu-latest
    ```
4. Prepare sample datasets, using the toy data in `AutoDL_sample_data` or download new datasets.

5. Run local test
```shell script
python run_local_test.py
```
The full usage is
```shell script
python run_local_test.py -dataset_dir='AutoDL_sample_data/miniciao' -code_dir='AutoDL_sample_code_submission'
```
Then you can view the real-time feedback with a learning curve by opening the
HTML page in `AutoDL_scoring_output/`.


Details can be seen in [AutoDL Challenge official starting_kit](https://github.com/zhengying-liu/autodl_starting_kit_stable).


# 10. Contributing

Feel free to dive in! [Open an issue](https://github.com/DeepWisdom/AutoDL/issues/new) or submit PRs.

# 11. Contact us
[![img](https://raw.githubusercontent.com/DeepWisdom/AutoDL/pip/assets/deepwisdom-logo-white.svg "title")](http://fuzhi.ai/)


# 12. Join the Community
Scan QR code and join AutoDL community!

<img src="https://raw.githubusercontent.com/DeepWisdom/AutoDL/pip/assets/WechatIMG15.png" width = "500" height = "200" alt="AutoDL Community" align=center />


# 13. License
[Apache License 2.0](https://github.com/DeepWisdom/AutoDL/blob/master/LICENSE)
