### 解决方案：

#### 一、线下做好的，把结果保留在docker镜像里，线上就不再执行（线上执行会出现爆内存的错误）
1、数据清洗：根据初赛的经验，对测试集直接筛选保留一些列；训练集取2018年5、6月的数据。结果存在user_data/tmp_data/文件夹里。

2、数据标注：根据fault_tag对训练集进行标注，将该盘报错时间-当前时间<30天的记录标为1,否则标为0。结果存在user_data/tmp_data/文件夹里。

#### 二、线上操作，就是提交docker镜像后在阿里云比赛平台执行的部分
3、将测试集按列进行数值统计，保留非空列及非唯一值列和一些后面会用到的列;训练集保留与测试集相同的列+label特征列。

4、特征提取：对1、5、7、199的raw值列，对每块盘取每条记录与前一天的差值;去掉10、240、241、242、1、195的raw值列，以及10、240、241、242、194、199的normalized值列

5、模型训练：将训练集和测试集分为两部分，标为正常集（5、187、188、197、198的normalized值全为100）和异常集（5、187、188、197、198的normalized值不全为100）;
   将正常集数据根据model==1或2，分成两部分数据，分别进行预测；
   对正常集建立LightGBM模型，采用五折交叉验证，并用五折模型的预测平均值作为测试集的测试结果，筛选出测试结果为1且5、187、188的raw值不全为0的记录;
   镜像：registry.cn-shanghai.aliyuncs.com/xl717/pakdd2020:2.3，就是对异常集筛选出5、187、188的raw值不全为0的记录;
   镜像：registry.cn-shanghai.aliyuncs.com/xl717/pakdd2020:2.4，就是对异常集筛选出5_raw>0 and 187、188的raw值不全为0的记录;
   最后将三种数据集提取出的结果合并，生成最终结果

### 结果：
镜像：registry.cn-shanghai.aliyuncs.com/xl717/pakdd2020:2.3
```buildoutcfg
score:23.5786
precision:23.0769
recall:24.1026
```

镜像：registry.cn-shanghai.aliyuncs.com/xl717/pakdd2020:2.4
```buildoutcfg
score:21.0237
precision:26.4706
recall:17.4359
```

### 生成镜像目录结构
```
project2
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
│       ├── disk_sample_smart_log_201806.csv
│       └── disk_sample_smart_log_201807.csv
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
├── prediction_result
├── run.sh
├── tcdata
│   └── disk_sample_smart_log_round2
└── user_data
    ├── .DS_Store
    └── tmp_data
        ├── .DS_Store
        ├── 201806.csv
        └── 201807.csv
```