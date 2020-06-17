# 走进AutoDL-音频分类系列

## 说话人确认实战

### AutoSpeech原始数据集链接：

- **百度盘:** [https://pan.baidu.com/s/1SbeamQQwTHUGmtP_qxt_bA](https://pan.baidu.com/s/1SbeamQQwTHUGmtP_qxt_bA) password:fu9p
- **Google Drive:**
  - (starting kit) [https://drive.google.com/drive/folders/16GCWu0JhkHiZ2mMuV_BtoIEF47xRXdpN?usp=sharing](https://drive.google.com/file/d/14xWg58IXXJZvypdEhTEWGBLs6O6VNMz8/view?usp=sharing)
  - (practice dataset) [https://drive.google.com/drive/folders/1jrxX9tlRx7WPYxLjSN7Vqnn0V1QE6bj6?usp=sharing](https://drive.google.com/drive/folders/1jrxX9tlRx7WPYxLjSN7Vqnn0V1QE6bj6?usp=sharing)

### 原始数据准备
下载`Ryerson Autdio-Visual`数据集中的Audio_Song_Actors[https://zenodo.org/record/1188976/files/Audio_Song_Actors_01-24.zip?download=1](https://zenodo.org/record/1188976/files/Audio_Song_Actors_01-24.zip?download=1)  
按下述方法准备数据：  

新建文件夹`SongActors`，并准备两个文件`labels.name`和`labels.csv`，并将音频文件放于该目录下。<br />`labels.name`为分类标签列表，每行一个标签，如：<br />

```
actor_00
actor_01
actor_02
```

<br />`labels.csv`为原始音频名与标签索引的对应表，分隔符为：`,`，两个列名为：`FileName`和`Labels`，如：<br />

```
FileName,Labels
03-02-01-01-01-01-01.wav,0
03-02-02-02-02-02-01.wav,0
03-02-04-01-02-01-01.wav,0
03-02-05-02-01-02-01.wav,0
```


### 标准数据集转换
使用autodl自带的数据转换器将Speech类数据转成autodl的tfrecords格式，样例代码如下：<br />参考 examples/data_convert_example.py

```python
from autodl.convertor import autospeech_2_autodl_format

def convertor_speech_demo():
    raw_autospeech_datadir = "~/AutoSpeech/AutoDL_sample_data/SongActors/"
    autospeech_2_autodl_format(input_dir=raw_autospeech_datadir)

convertor_speech_demo()    
```

执行后得到autodl tfrecords的数据集SongActors_formatted如下：
```
├── SongActors_formatted
│   ├── SongActors_formatted.data
│   │   ├── test
│   │   │   ├── metadata.textproto												# 测试集元数据
│   │   │   └── sample-SongActors_formatted-test.tfrecord			            # 测试集数据
│   │   └── train
│   │       ├── metadata.textproto												# 训练集元数据
│   │       └── sample-SongActors_formatted-train.tfrecord		                # 训练集数据及标签
│   └── SongActors_formatted.solution											# 测试集Label
```

### 自动训练预测和评估
```python
import os
import argparse
import time

from autodl.convertor.speech_to_tfrecords import autospeech_2_autodl_format
from autodl.auto_ingestion import data_io
from autodl.auto_ingestion.dataset import AutoDLDataset
from autodl.auto_models.at_speech.model import Model as SpeechModel
from autodl.utils.util import get_solution
from autodl.metrics import autodl_auc, accuracy


def run_single_model(model, dataset_dir, basename, time_budget=1200, max_epoch=50):
    D_train = AutoDLDataset(os.path.join(dataset_dir, basename, "train"))
    D_test = AutoDLDataset(os.path.join(dataset_dir, basename, "test"))
    solution = get_solution(solution_dir=dataset_dir)

    start_time = int(time.time())
    for i in range(max_epoch):
        remaining_time_budget = start_time + time_budget - int(time.time())
        model.fit(D_train.get_dataset(), remaining_time_budget=remaining_time_budget)

        remaining_time_budget = start_time + time_budget - int(time.time())
        y_pred = model.predict(D_test.get_dataset(), remaining_time_budget=remaining_time_budget)

        # Evaluation.
        nauc_score = autodl_auc(solution=solution, prediction=y_pred)
        acc_score = accuracy(solution=solution, prediction=y_pred)

        print("Epoch={}, evaluation: nauc_score={}, acc_score={}".format(i, nauc_score, acc_score))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="speech example arguments")
    parser.add_argument("--input_data_path", type=str, help="path of input data")
    args = parser.parse_args()

    input_dir = os.path.dirname(args.input_data_path)

    autospeech_2_autodl_format(input_dir=input_dir)

    new_dataset_dir = input_dir + "_formatted" + "/" + os.path.basename(input_dir)
    datanames = data_io.inventory_data(new_dataset_dir)
    basename = datanames[0]
    print("train_path: ", os.path.join(new_dataset_dir, basename, "train"))

    D_train = AutoDLDataset(os.path.join(new_dataset_dir, basename, "train"))

    max_epoch = 50
    time_budget = 1200

    model = SpeechModel(D_train.get_metadata())

    run_single_model(model, new_dataset_dir, basename, time_budget, max_epoch)

```
上述代码中 `y_pred` 对测试数据集的预测结果，输出样本对应每个label的概率。<br />评估方式中 nauc_score 为正则化后auc分数 2*auc - 1，acc_score 为准确率。<br />



