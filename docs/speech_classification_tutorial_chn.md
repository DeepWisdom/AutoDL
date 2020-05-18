# 走进AutoDL-音频分类系列

## 说话人确认实战

### AutoSpeech原始数据集链接：

- **百度盘:** [https://pan.baidu.com/s/1SbeamQQwTHUGmtP_qxt_bA](https://pan.baidu.com/s/1SbeamQQwTHUGmtP_qxt_bA) password:fu9p
- **Google Drive:**
  - (starting kit) [https://drive.google.com/drive/folders/16GCWu0JhkHiZ2mMuV_BtoIEF47xRXdpN?usp=sharing](https://drive.google.com/file/d/14xWg58IXXJZvypdEhTEWGBLs6O6VNMz8/view?usp=sharing)
  - (practice dataset) [https://drive.google.com/drive/folders/1jrxX9tlRx7WPYxLjSN7Vqnn0V1QE6bj6?usp=sharing](https://drive.google.com/drive/folders/1jrxX9tlRx7WPYxLjSN7Vqnn0V1QE6bj6?usp=sharing)

### 原始数据准备
我们以AutoSpeech竞赛说话人确认数据集data01为例，数据集目录结构如下所示：
> data01
> ├── data01.data
> │   ├── meta.json             //数据集元数据
> │   ├── test.pkl                //测试集语音语料
> │   ├── train.pkl              //训练集语音语料
> │   └── train.solution        //训练集Label标签
> └── data01.solution             //测试集Label标签               

```json
{
  	"class_num": 100, 					# 音频分类类别数
  	"train_num": 3000, 					# 训练集样本数
  	"test_num": 3000, 					# 测试样本数
  	"time_budget": 1800					# 训练及测试时长限制，单位秒
}
```


### 标准数据集转换
使用autodl自带的数据转换器将Speech类数据转成autodl的tfrecords格式，样例代码如下：<br />参考 examples/data_convert_example.py

```python
from autodl.convertor import autospeech_2_autodl_format

def convertor_speech_demo():
    raw_autospeech_datadir = "~/AutoSpeech/AutoDL_sample_data/data01"
    autospeech_2_autodl_format(input_dir=raw_autospeech_datadir)

convertor_speech_demo()    
```

执行后得到autodl tfrecords的数据集data01_formatted如下：
```json
├── data01_formatted
│   ├── data01_formatted.data
│   │   ├── test
│   │   │   ├── metadata.textproto												# 测试集元数据
│   │   │   └── sample-data01_formatted-test.tfrecord			                # 测试集数据
│   │   └── train
│   │       ├── metadata.textproto												# 训练集元数据
│   │       └── sample-data01_formatted-train.tfrecord		                    # 训练集数据及标签
│   └── data01_formatted.solution												# 测试集Label
```

### 自动训练预测和评估
```python
import os

from autodl import Model, AutoDLDataset
from autodl.auto_ingestion import dataset_utils_v2
from autodl.auto_scoring.score import get_solution
from autodl.metrics import autodl_auc, accuracy

def do_speech_classification_demo():
    remaining_time_budget = 1200
    max_epoch = 100

    # Speech autodl format tfrecords
    dataset_dir = "ADL_sample_data/data01_formatted"

    basename = dataset_utils_v2.get_dataset_basename(dataset_dir)
    D_train = AutoDLDataset(os.path.join(dataset_dir, basename, "train"))
    D_test = AutoDLDataset(os.path.join(dataset_dir, basename, "test"))
    solution = get_solution(solution_dir=dataset_dir)

    M = Model(D_train.get_metadata())  # The metadata of D_train and D_test only differ in sample_count

    for i in range(max_epoch):
        M.fit(D_train.get_dataset(), remaining_time_budget)
		
        # Y_pred shape=(test_num, class_num), probability of labels
        Y_pred = M.predict(D_test.get_dataset(), remaining_time_budget)

        # Evaluation.
        nauc_score = autodl_auc(solution=solution, prediction=Y_pred)
        acc_score = accuracy(solution=solution, prediction=Y_pred)

        print(Y_pred)
        print("Epoch={}, evaluation: nauc_score={}, acc_score={}".format(i, nauc_score, acc_score))


def main():
    do_speech_classification_demo()


if __name__ == '__main__':
    main()
```
上述代码中 `y_pred` 对测试数据集的预测结果，输出样本对应每个label的概率。<br />评估方式中 nauc_score 为正则化后auc分数 2*auc - 1，acc_score 为准确率。<br />



