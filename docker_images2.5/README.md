### 解决方案：

#### 一、线下做好的，把结果保留在docker镜像里，线上就不再执行（线上执行会出现爆内存的错误）
1、数据清洗：根据初赛的经验，对测试集直接筛选保留一些列；训练集取2018年7、8月的数据。结果存在user_data/tmp_data/文件夹里。

2、数据标注：根据fault_tag对训练集进行标注，将['model', 'serial_number']在fault_tag中存在的盘记录标为1,否则标为0，只保留每个盘当月的最后一条记录。
   结果存在user_data/tmp_data/文件夹里。

#### 二、线上操作，就是提交docker镜像后在阿里云比赛平台执行的部分
3、先将测试集排序后去重，只留每个盘整个月份的最后一个数据，再对测试集按列进行数值统计，保留非空列及非唯一值列和一些后面会用到的列;
   训练集保留与测试集相同的列+label特征列。

4、特征提取：对1、5、7、199的raw值列，对每块盘取每条记录与前一天的差值;去掉10、240、241、242、1、195的raw值列，以及10、240、241、242、194、199的normalized值列

5、模型训练：将数据根据model==1或2，分成两部分数据，分别进行预测；
   将每一部分数据正负样本按照1：10的比例，对负样本采取随机采样， 
   建立LightGBM模型，采用五折交叉验证，并用五折模型的预测平均值作为测试集的测试结果，
   最后将model==1或2的两个的结果合并，生成最终结果

### 结果：
镜像：registry.cn-shanghai.aliyuncs.com/xl717/pakdd2020:2.5
```buildoutcfg
score:0
precision:0
recall:0
```

### 生成镜像目录结构
```
project2
├── code
│   ├── .DS_Store
│   └── main.py
├── data
│   ├── .DS_Store
│   └── round2_train
│       ├── disk_sample_fault_tag_201808.csv
│       ├── disk_sample_fault_tag.csv
│       ├── disk_sample_smart_log_201807.csv
│       ├── disk_sample_smart_log_201808.csv
│       └── .DS_Store
├── Dockerfile
├── .DS_Store
├── feature
│   ├── data_cleaning.py
│   ├── .DS_Store
│   ├── feature.py
│   ├── label.py
│   └── __pycache__
│       ├── data_cleaning.cpython-36.pyc
│       ├── data_cleaning.cpython-37.pyc
│       ├── feature.cpython-36.pyc
│       ├── feature.cpython-37.pyc
│       ├── label.cpython-36.pyc
│       └── label.cpython-37.pyc
├── .idea
│   ├── inspectionProfiles
│   │   └── profiles_settings.xml
│   ├── misc.xml
│   ├── modules.xml
│   ├── project2.iml
│   └── workspace.xml
├── model
│   ├── basic_model.py
│   └── __pycache__
│       ├── basic_model.cpython-36.pyc
│       └── basic_model.cpython-37.pyc
├── prediction_result
├── README.md
├── run.sh
├── tcdata
│   └── disk_sample_smart_log_round2
└── user_data
    ├── .DS_Store
    └── tmp_data
        ├── 201807.csv
        ├── 201808.csv
        └── .DS_Store

14 directories, 34 files

```
