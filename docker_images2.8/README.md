### 解决方案：

#### 一、线下做好的，把结果保留在docker镜像里，线上就不再执行（线上执行会出现爆内存的错误）
1、数据清洗：根据初赛的经验，对测试集直接筛选保留一些列；训练集取2017年7-12月,2018年1-8月的数据。结果存在user_data/tmp_data/文件夹里。

2、数据标注：根据fault_tag对训练集进行标注，将['model', 'serial_number']在fault_tag中存在的盘的每个月的最后一条记录标为1,其他的都标为0。
   结果存在user_data/tmp_data/文件夹里。
#### 二、线上操作，就是提交docker镜像后在阿里云比赛平台执行的部分
3、对测试集按列进行数值统计，保留非空列及非唯一值列和一些后面会用到的列;
   训练集保留与测试集相同的列+label特征列。

4、去掉10、240、241、242、1、195的raw值列，\
    以及10、240、241、242、194、199的normalized值列

5、根据可调参数做数据筛选和特征工程
# 可调节的参数
# 特征工程：是否构建diff特征，是否新增raw/normalized列，是否新增raw/normalized的diff特征，是：1，否：0
# is_raw_normalized_diff在is_raw_normalized为1的前提下，才可以赋值为1
is_getdiff, is_raw_normalized, is_raw_normalized_diff = 1, 1, 1
# 特征工程：是否保留raw特征，是否保留normalized列，是否保留diff特征，是：1，否：0
# is_deldiff在is_getdiff，is_raw_normalized_diff至少有一个为1的前期下，才可以赋值为1
is_delraw, is_delnormalized, is_deldiff = 1, 1, 1
# 数据筛选：训练集中负样本只保留正样本的前20天数据，测试集中样本只保留样本的最近21天数据，同为1或着同为0，测试集和训练集保持一致
# 数据筛选必须在diff构建之后执行
is_data_neg, is_test21 = 1, 1
# lightgbm模型的折数和预测阈值
n_splits, pred_threshold = 10, 0.5

6、模型训练：
   将数据正负样本按照1：10的比例，对负样本采取随机采样， 
   建立LightGBM模型，采用设定的折数折交叉验证，并用预测平均值>=预测阈值作为测试集的测试结果

### 结果：
镜像：registry.cn-shanghai.aliyuncs.com/xl717/pakdd2020:2.8
```buildoutcfg
score:
precision:
recall:
```

### 生成镜像目录结构
```
.
├── .DS_Store
├── .idea
│   ├── inspectionProfiles
│   │   └── profiles_settings.xml
│   ├── misc.xml
│   ├── modules.xml
│   ├── project2.iml
│   └── workspace.xml
├── Dockerfile
├── README.md
├── code
│   ├── .DS_Store
│   └── main.py
├── data
│   ├── .DS_Store
│   └── round2_train
│       ├── .DS_Store
│       ├── disk_sample_fault_tag.csv
│       ├── disk_sample_fault_tag_201808.csv
│       ├── disk_sample_smart_log_201707.csv
│       ├── disk_sample_smart_log_201708.csv
│       ├── disk_sample_smart_log_201709.csv
│       ├── disk_sample_smart_log_201710.csv
│       ├── disk_sample_smart_log_201711.csv
│       ├── disk_sample_smart_log_201712.csv
│       ├── disk_sample_smart_log_201801.csv
│       ├── disk_sample_smart_log_201802.csv
│       ├── disk_sample_smart_log_201803.csv
│       ├── disk_sample_smart_log_201804.csv
│       ├── disk_sample_smart_log_201805.csv
│       ├── disk_sample_smart_log_201806.csv
│       ├── disk_sample_smart_log_201807.csv
│       └── disk_sample_smart_log_201808.csv
├── feature
│   ├── .DS_Store
│   ├── __pycache__
│   │   ├── data_cleaning.cpython-36.pyc
│   │   ├── data_cleaning.cpython-37.pyc
│   │   ├── feature.cpython-36.pyc
│   │   ├── feature.cpython-37.pyc
│   │   ├── label.cpython-36.pyc
│   │   └── label.cpython-37.pyc
│   ├── data_cleaning.py
│   ├── feature.py
│   └── label.py
├── model
│   ├── __pycache__
│   │   ├── basic_model.cpython-36.pyc
│   │   └── basic_model.cpython-37.pyc
│   └── basic_model.py
├── prediciton_result
├── run.sh
├── tcdata
│   ├── .DS_Store
│   └── disk_sample_smart_log_round2
└── user_data
    ├── .DS_Store
    └── tmp_data
        ├── .DS_Store
        ├── 201707.csv
        ├── 201708.csv
        ├── 201709.csv
        ├── 201710.csv
        ├── 201711.csv
        ├── 201712.csv
        ├── 201801.csv
        ├── 201802.csv
        ├── 201803.csv
        ├── 201804.csv
        ├── 201805.csv
        ├── 201806.csv
        ├── 201807.csv
        └── 201808.csv

14 directories, 59 files

```