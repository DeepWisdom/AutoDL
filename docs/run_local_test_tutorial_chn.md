# 走进AutoDL之任意模态动态效果查看


## 公共操作
```shell script
pip install autodl-gpu==1.0.0
```

## 数据集准备
AutoDL竞赛TFRecords格式数据集下载页面，见 [https://autodl.lri.fr/competitions/162#learn_the_details-get_data](https://autodl.lri.fr/competitions/162#learn_the_details-get_data]).

## 数据集查看
以表格Tabular-Madline数据集为例，结构为，
```shell script
madeline/
├── madeline.data
│   ├── test
│   │   ├── metadata.textproto
│   │   └── sample-madeline-test.tfrecord
│   └── train
│       ├── metadata.textproto
│       └── sample-madeline-train.tfrecord
└── madeline.solution
```

## 本地运行测试
```shell script
python run_local_test.py --dataset_dir=../adl_sample_data/madeline --output_dir=out_madline
```

- 本地实时效果查看
效果目录结构为，
```shell script
out_madline
├── end.txt
├── learning-curve-madeline.png
└── scores.txt
```





