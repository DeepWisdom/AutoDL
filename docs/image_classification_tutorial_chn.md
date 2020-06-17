# 走进AutoDL-图像分类系列

<a name="f3da820f"></a>
# 公共操作

<br />可以用pip安装autodl的第一个稳定版本。使用命令行环境用下面代码单元安装autodl<br />

```
pip install autodl-gpu
```

<br />pip安装会自动安装所有依赖项，完整依赖项列表参照下方链接：[https://github.com/DeepWisdom/AutoDL/blob/pip/requirements.txt](https://github.com/DeepWisdom/AutoDL/blob/pip/requirements.txt)

<a name="3f54a3bd"></a>
# 使用AutoDL进行monkeys种类图像分类问题


<a name="52f97eb8"></a>
## audodl入门

<br />autodl安装见上述公共操作。<br />

<a name="9a0ca25d"></a>
## 数据集准备
- Image样例数据集-monkeys.zip

百度云盘链接:https://pan.baidu.com/s/1OAbn9p7PbIhNYMJM0UHEQA  密码:bkhs

<br />新建文件夹`monkeys`，并准备两个文件`labels.name`和`labels.csv`，并将图像文件放于该目录下。<br />`labels.name`为分类标签列表，每行一个标签，如：<br />

```
Baboon
Chimp
Gorilla
```

<br />`labels.csv`为原始图片名与标签索引的对应表，分隔符为：`,`，两个列名为：`FileName`和`Labels`，如：<br />

```
FileName,Labels
n7031.jpg,7
n7145.jpg,7
n0159.jpg,0
```


<a name="f1f3d545"></a>
## 标准数据集转换

<br />使用`autodl`自带的数据转换器将原始图片格式转为训练需要的`tfrecords`格式，示例代码如下：<br />

```python
from autodl.convertor.image_to_tfrecords import autoimage_2_autodl_format

def convertor_image_demo():
    raw_autoimage_datadir = f"{path}/monkeys/"
    autoimage_2_autodl_format(input_dir=raw_autoimage_datadir)

convertor_image_demo()
```

<br />执行后得到的`image`的`tfrecords`的数据集格式如下：<br />

```json
.../monkeys_formatted/
└── monkeys
    ├── monkeys.data
    │   ├── test                                  # 测试集
    │   │   ├── metadata.textproto                # 测试集元数据
    │   │   └── sample-monkeys-test.tfrecord      # 测试集数据
    │   └── train                                 # 训练集
    │       ├── metadata.textproto                # 训练集元数据
    │       └── sample-monkeys-train.tfrecord     # 训练集数据集标签
    └── monkeys.solution
```


<a name="3e8da0a7"></a>
## 训练和评估

<br />使用下述代码进行训练和评估<br />

```python
import os
import time
import argparse

from autodl.auto_ingestion import data_io
from autodl.utils.util import get_solution
from autodl.metrics import autodl_auc
from autodl.auto_ingestion.dataset import AutoDLDataset
from autodl.auto_models.auto_image.model import Model as ImageModel

from autodl import AutoDLDataset
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
    parser = argparse.ArgumentParser(description="tabular example arguments")
    parser.add_argument("--input_data_path", type=str, help="path of input data")
    args = parser.parse_args()

    input_dir = os.path.dirname(args.input_data_path)

    autoimage_2_autodl_format(input_dir=input_dir)

    new_dataset_dir = input_dir + "_formatted" + "/" + os.path.basename(input_dir)
    datanames = data_io.inventory_data(new_dataset_dir)
    basename = datanames[0]
    print("train_path: ", os.path.join(new_dataset_dir, basename, "train"))

    D_train = AutoDLDataset(os.path.join(new_dataset_dir, basename, "train"))
    D_test = AutoDLDataset(os.path.join(new_dataset_dir, basename, "test"))

    max_epoch = 50
    time_budget = 1200

    model = ImageModel(D_train.get_metadata())

    run_single_model(model, new_dataset_dir, basename, time_budget, max_epoch)
```


