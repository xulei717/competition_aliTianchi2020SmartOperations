### 解决方案：

#### 一、线下做好的，把结果保留在docker镜像里，线上就不再执行（线上执行会出现爆内存的错误）
1、数据清洗：根据初赛的经验，对测试集直接筛选保留一些列；训练集取2017年7-12月,2018年1-8月的数据。结果存在user_data/tmp_data/文件夹里。

2、数据标注：根据fault_tag对训练集进行标注，将['model', 'serial_number']在fault_tag中存在的盘的每个月的最后一条记录标为1,其他的都标为0。
   结果存在user_data/tmp_data/文件夹里。

#### 二、线上操作，就是提交docker镜像后在阿里云比赛平台执行的部分
3、对测试集按列进行数值统计，保留非空列及非唯一值列和一些后面会用到的列;
   训练集保留与测试集相同的列+label特征列。

4、特征提取：对1、5、7、199的raw值列，对每块盘取每条记录与前一天的差值;

5、去掉10、240、241、242、1、195的raw值列，\
    以及10、240、241、242、194、199的normalized值列

6、模型训练：
   将数据正负样本按照1：10的比例，对负样本采取随机采样， 
   建立LightGBM模型，采用10折交叉验证，并用预测平均值>=0.1作为测试集的测试结果

### 结果：
镜像：registry.cn-shanghai.aliyuncs.com/xl717/pakdd2020:2.7
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