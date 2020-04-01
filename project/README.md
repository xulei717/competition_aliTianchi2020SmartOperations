### 解决方案：
```
1、数据清洗：将测试集按列进行数值统计，保留非空列及非唯一值列;训练集取2018年5、6月的数据，保留与测试集相同的列
2、数据标注：根据fault_tag对训练集进行标注，将该盘报错时间-当前时间<30天的记录标为1,否则标为0
3、特征提取：对1、5、7、199的raw值列，对每块盘取每条记录与前一天的差值;去掉10、240、241、242、1、195的raw值列，以及10、240、241、242、194、199的normalized值列
4、模型训练：将训练集和测试集分为两部分，标为正常集（5、187、188、197、198的normalized值全为100）和异常集（5、187、188、197、198的normalized值不全为100）;对正常集建立LightGBM模型，采用五折交叉验证，并用五折模型的预测平均值作为测试集的测试结果，筛选出测试结果为1且5、187、188的raw值不全为0的记录;对异常集筛选出5、187、188的raw值不全为0的记录;最后将两种数据集提取出的结果合并，生成最终结果
```

### 运行说明：
```
在project/code/文件夹下执行 python main.py
```

### 文件夹结构
```
.
├── code
│   ├── main.py
│   └── requirements.txt
├── data
│   ├── round1_testA
│   ├── round1_testB
│   │   └── disk_sample_smart_log_test_b.csv
│   └── round1_train
│       ├── disk_sample_fault_tag.csv
│       ├── disk_sample_smart_log_201805.csv
│       └── disk_sample_smart_log_201806.csv
├── feature
│   ├── data_cleaning.py
│   ├── feature.py
│   ├── label.py
│   └── __pycache__
│       ├── data_cleaning.cpython-36.pyc
│       ├── feature.cpython-36.pyc
│       └── label.cpython-36.pyc
├── .idea
│   ├── misc.xml
│   ├── modules.xml
│   ├── project.iml
│   └── workspace.xml
├── model
│   ├── basic_model.py
│   └── __pycache__
│       ├── basic_model.cpython-36.pyc
│       └── basic_model.cpython-37.pyc
├── prediction_result
│   └── predictions.csv
├── README.md
└── user_data
    └── tmp_data
        ├── 201805.csv
        ├── 201806.csv
        └── test_b.csv

13 directories, 24 files

```
