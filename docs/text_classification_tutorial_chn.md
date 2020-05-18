# 使用AutoDL进行电影评论分类实战

## autodl入门
- 安装：`pip install autodl-gpu`

## AutoNLP原始数据集链接：

- **百度盘**：[https://pan.baidu.com/s/1Snl0vkkZpmMJU8VNgU598w](https://pan.baidu.com/s/1Snl0vkkZpmMJU8VNgU598w), 密码：zmwq
- **GoogleDrive**: [https://drive.google.com/open?id=1X6N-1a5h78G9M6OErPELRX2nIE8vZltH](https://drive.google.com/open?id=1X6N-1a5h78G9M6OErPELRX2nIE8vZltH)

## 原始数据准备
我们以AutoNLP竞赛数据集O1为例，数据集目录结构如下所示：
> ├── O1.data
> │   ├── meta.json             //数据集元数据
> │   ├── test.data              //测试集文本语料
> │   ├── train.data             //训练集文本语料
> │   └── train.solution        //训练集Label标签
> └── O1.solution                 //测试集Label标签

元数据meta.json说明:
> {
>    "class_num": 2,
>    "train_num": 7792,
>    "test_num": 1821,
>    "language": "EN",
>    "time_budget": 2400
> }

```json
{
    "class_num": 2,							# 文本分类类别数
    "train_num": 7792,					    # 训练集样本数
    "test_num": 1821,						# 测试集样本数
    "language": "EN",						# 文本语言类型
    "time_budget": 2400				        # 训练及测试时长限制，单位秒
}
```

## 标准数据集转换
使用autodl自带的数据转换器将Text类数据转成autodl的tfrecords格式，样例代码如下：参考 examples/data_convert_example.py

```python
from autodl.convertor import autonlp_2_autodl_format

def convertor_nlp_demo():
    raw_autonlp_datadir = "~/AutoNLP/AutoDL_sample_data/O1"
    autonlp_2_autodl_format(input_dir=raw_autonlp_datadir)

convertor_nlp_demo()    
```
执行后得到autodl tfrecords的数据集O1_formatted如下：
```json
├── O1_formatted
│   ├── O1_formatted.data
│   │   ├── test
│   │   │   ├── metadata.textproto										# 测试集元数据及词表
│   │   │   └── sample-O1_formatted-test.tfrecord			            # 测试集数据及标签
│   │   └── train
│   │       ├── metadata.textproto										# 训练集元数据及词表
│   │       └── sample-O1_formatted-train.tfrecord		                # 训练集数据及标签
│   └── O1_formatted.solution										    # 测试集Label
```
### 自动训练预测和评估
```python
import os

from autodl import Model, AutoDLDataset
from autodl.auto_ingestion import dataset_utils_v2
from autodl.auto_scoring.score import get_solution
from autodl.metrics import autodl_auc, accuracy

def do_text_classification_demo():
    remaining_time_budget = 1200
    max_epoch = 100

    # Text autodl format tfrecords
    dataset_dir = "ADL_sample_data/O1_formatted"

    basename = dataset_utils_v2.get_dataset_basename(dataset_dir)
    D_train = AutoDLDataset(os.path.join(dataset_dir, basename, "train"))
    D_test = AutoDLDataset(os.path.join(dataset_dir, basename, "test"))
    solution = get_solution(solution_dir=dataset_dir)

    M = Model(D_train.get_metadata())  # The metadata of D_train and D_test only differ in sample_count

    for i in range(max_epoch):
        M.fit(D_train.get_dataset())
		
        # Y_pred shape=(test_num, class_num), probability of labels
        Y_pred = M.predict(D_test.get_dataset())

        # Evaluation.
        nauc_score = autodl_auc(solution=solution, prediction=Y_pred)
        acc_score = accuracy(solution=solution, prediction=Y_pred)

        print(Y_pred)
        print("Epoch={}, evaluation: nauc_score={}, acc_score={}".format(i, nauc_score, acc_score))


def main():
    do_text_classification_demo()


if __name__ == '__main__':
    main()
```
上述代码中 `y_pred` 对测试数据集的预测结果，输出样本对应每个label的概率。<br />评估方式中 nauc_score 为正则化后auc分数 2*auc - 1，acc_score 为准确率。<br />

