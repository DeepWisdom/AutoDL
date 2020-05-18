# 走进AutoDL-视频分类系列

<a name="f3da820f"></a>
# 公共操作

<br />可以用pip安装autodl的第一个稳定版本(autodl 1.0.0)。使用命令行环境用下面代码单元安装autodl<br />

```
pip install autodl-gpu
```

<br />pip安装会自动安装所有依赖项，完整依赖项列表参照下方链接：<br />`[https://github.com/DeepWisdom/AutoDL/blob/pip/requirements.txt](https://github.com/DeepWisdom/AutoDL/blob/pip/requirements.txt)`<br />

<a name="dd54a85a"></a>
# 使用AutoDL进行mini-kth人体行为识别分类问题


<a name="52f97eb8"></a>
## audodl入门

<br />autodl安装见上述公共操作。<br />

<a name="9a0ca25d"></a>
## 数据集准备
- Video 样例数据集 mini-kth.zip
百度云盘链接:https://pan.baidu.com/s/1OAbn9p7PbIhNYMJM0UHEQA  密码:bkhs



<br />新建文件夹`mini-kth`，并准备两个文件`label.name`和`labels.csv`。<br />`labels.name`为分类标签列表，每行一个标签，如：<br />

```
running
walking
```

<br />`labels.csv`为原始视频名与标签索引的对应表，分隔符为：`,`，两个列名为：`FileName`和`Labels`，如：<br />

```
FileName,Labels
video1.avi,3
video2.avi,2
```


<a name="f1f3d545"></a>
## 标准数据集转换

<br />使用`autodl`自带的数据转换器将原始视频格式转为训练需要的`tfrecords`格式，示例代码如下：<br />

```
from autodl.convertor.video_to_tfrecords import autovideo_2_autodl_format

def convertor_video_demo():
    raw_autovideo_datadir = f"{path}/mini-kth/"
    autovideo_2_autodl_format(input_dir=raw_autovideo_datadir)

convertor_video_demo()
```

<br />执行后得到的`video`的`tfrecords`的数据集格式如下：<br />

```
.../mini-kth/
├── mini-kth.data
│   ├── test                                      # 测试集
│   │   ├── metadata.textproto                    # 测试集元数据
│   │   └── sample-mini-kth-test.tfrecord         # 测试集数据
│   └── train                                     # 训练集
│       ├── metadata.textproto                    # 训练集元数据
│       └── sample-mini-kth-train.tfrecord        # 训练集数据集标签
└── mini-kth.solution
```


<a name="3e8da0a7"></a>
## 训练和评估

<br />使用下述代码进行训练和评估<br />

```
import os
import time
import argparse

from autodl.convertor.video_to_tfrecords import autovideo_2_autodl_format
from autodl.auto_ingestion import data_io
from autodl.auto_ingestion.dataset import AutoDLDataset
from autodl.auto_models.auto_video.model import Model as VideoModel
from autodl.auto_ingestion.pure_model_run import run_single_model

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
    parser = argparse.ArgumentParser(description="video example arguments")
    parser.add_argument("--input_data_path", type=str, help="path of input data")
    args = parser.parse_args()

    input_dir = os.path.dirname(args.input_data_path)

    autovideo_2_autodl_format(input_dir=input_dir)

    new_dataset_dir = input_dir + "_formatted" + "/" + os.path.basename(input_dir)
    datanames = data_io.inventory_data(new_dataset_dir)
    basename = datanames[0]
    print("train_path: ", os.path.join(new_dataset_dir, basename, "train"))

    D_train = AutoDLDataset(os.path.join(new_dataset_dir, basename, "train"))
    D_test = AutoDLDataset(os.path.join(new_dataset_dir, basename, "test"))

    max_epoch = 50
    time_budget = 1200

    model = VideoModel(D_train.get_metadata())

    run_single_model(model, new_dataset_dir, basename, time_budget, max_epoch)
```


