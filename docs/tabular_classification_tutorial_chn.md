# 使用AutoDL-表格分类-进行银行定期存储表格分类

## 安装
```
pip install autodl-gpu
```

## audodl入门
autodl安装见上述公共操作。

## 数据集准备

数据集上使用kaggle上的`[https://www.kaggle.com/henriqueyamahata/bank-marketing](https://www.kaggle.com/henriqueyamahata/bank-marketing)`的`bank-additional-full.csv`数据表，并进行随机切分为训练和测试集。
对于数据集中的类型特征，先使用`LabelEncoder`进行预处理下。

## 标准数据集转换

使用`autodl`自带的数据转换器将原始表格数据转为训练需要的`tfrecords`格式，示例代码如下：

```@python
from autodl.convertor.tabular_to_tfrecords import autotabular_2_autodl_format

def convertor_tabular_demo():
    raw_autoimage_datadir = f"{path}/bank/bank-additional-full.csv"
    autotabular_2_autodl_format(input_dir=raw_autonlp_datadir)

convertor_tabular_demo()
```

执行后得到的`tabular`的`tfrecords`的数据集格式如下：

```
.../bank_formatted/
└── bank
    ├── bank.data
    │   ├── test                                  # 测试集
    │   │   ├── metadata.textproto                # 测试集元数据
    │   │   └── sample-bank-test.tfrecord         # 测试集数据
    │   └── train                                 # 训练集
    │       ├── metadata.textproto                # 训练集元数据
    │       └── sample-bank-train.tfrecord        # 训练集数据集标签
    └── bank.solution
```

## 训练和评估

使用下述代码进行训练和评估

```@python

import os
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from autodl.auto_ingestion import data_io
from autodl.utils.util import get_solution
from autodl.metrics import autodl_auc
from autodl.auto_ingestion.dataset import AutoDLDataset
from autodl.convertor.tabular_to_tfrecords import autotabular_2_autodl_format
from autodl.auto_models.auto_tabular.model import Model as TabularModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tabular example arguments")
    parser.add_argument("--input_data_path", type=str, help="path of input data")
    args = parser.parse_args()

    input_dir = os.path.dirname(args.input_data_path)

    df = pd.read_csv(args.input_data_path, sep=";")

    trans_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week",
                  "poutcome", "y"]
    for col in trans_cols:
        lbe = LabelEncoder()
        df[col] = lbe.fit_transform(df[col])

    label = df["y"]

    autotabular_2_autodl_format(input_dir=input_dir, data=df, label=label)

    new_dataset_dir = input_dir + "_formatted" + "/" + os.path.basename(input_dir)
    datanames = data_io.inventory_data(new_dataset_dir)
    basename = datanames[0]
    print("train_path: ", os.path.join(new_dataset_dir, basename, "train"))

    D_train = AutoDLDataset(os.path.join(new_dataset_dir, basename, "train"))
    D_test = AutoDLDataset(os.path.join(new_dataset_dir, basename, "test"))

    max_epoch = 100
    model = TabularModel(D_train.get_metadata())

    for i in range(max_epoch):
        model.fit(D_train.get_dataset())
        y_pred = model.predict(D_test.get_dataset())

        solution = get_solution(new_dataset_dir)

        nauc = autodl_auc(solution, y_pred)
        print(f"epoch: {i}, nauc: {nauc}")
```
