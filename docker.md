### 一、测试镜像能在本地跑通并成功推送到阿里云端
1.构建四个文件：Dockerfile, hello_world.py, result.json, run.sh

2.构建镜像:在四个文件目录下
```bash
docker build -t registry.cn-shanghai.aliyuncs.com/test717/tianchi_submit:1.0 .
```
运行镜像: docker run image的id或镜像名 sh run.sh
```bash
docker run registry.cn-shanghai.aliyuncs.com/test717/tianchi_submit:1.0 sh run.sh
```
运行成功输出一行信息: hello world!!!

如果要手动运行，创建好镜像后，创建一个容器并进入命令行系统运行代码（name你随便起）：
```bash
docker run -it --name=c_tianchi 镜像id /bin/bash
```
输入完成后就进入这个系统里了，下面是linux命令行：
查看文件信息
ls
运行shell脚本
```bash
sh run.sh
```
这样也能进行测试。


运行无误后，登录阿里云：
```bash
docker login --username=你的阿里云用户名 registry.cn-shanghai.aliyuncs.com
```
然后输入密码，即可返回登录成功的信息。

登录成功后可以推送镜像到云端（注意没登陆会被拒绝掉）：
```bash
docker push registry.cn-shanghai.aliyuncs.com/test717/tianchi_submit:1.0
```
这样就推送成功了，第一步完成！


### 二、project
1.构建第一层镜像：在project目录下,根据Dockerfile: python3镜像+ADD文件：registry.cn-shanghai.aliyuncs.com/xl717/pakdd2020:0325.0
```
sudo docker build -t registry.cn-shanghai.aliyuncs.com/xl717/pakdd2020:0325.0 .
```
Dockerfile文件内容改为：
```Dockerfile
# Base Images
## 从天池基础镜像构建
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/python:3

## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /
```
2.构建第二层镜像：执行镜像,再此s基础上安装包,生成新的镜像：python3镜像+ADDD文件+pip安装包：registry.cn-shanghai.aliyuncs.com/xl717/pakdd2020:0325.1
```bash
xulei$ sudo docker run -t -i registry.cn-shanghai.aliyuncs.com/xl717/pakdd2020:0325.0 /bin/bash
root@6cef26493644# pip install --upgrade pip
root@6cef26493644# pip install -r code/requirements.txt
root@6cef26493644# exit  ## 退出
```
更新镜像，产生新的镜像，通过命令docker commit提交容器副本
```bash
xulei$ docker commit -m="has update" -a="xulei" 6cef26493644 registry.cn-shanghai.aliyuncs.com/xl717/pakdd2020:0325.1
参数：
-6cef26493644：是容器ID
-registry.cn-shanghai.aliyuncs.com/xl717/pakdd2020:0325.1：指定要创建的目标镜像名
```

3.构建第三层镜像：在第二层镜像的基础上加上最后的执行命令：python3镜像+ADDD文件+pip安装包+CMD命令：registry.cn-shanghai.aliyuncs.com/xl717/pakdd2020:0325.3
文件结构为：
```buildoutcfg
.
├── Dockerfile
└── code
    └── requirements.txt
```
Dockerfile文件内容改为：
```Dockerfile
# Base Images
## 从天池基础镜像构建
FROM registry.cn-shanghai.aliyuncs.com/xl717/pakdd2020:0325.1

## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /

## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /

## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]
```
构建镜像:在一个新的文件夹下
```bash
xulei$ docker build -t registry.cn-shanghai.aliyuncs.com/xl717/pakdd2020:0325.2 .
```


### 三、project
新思路：docker共有两层，第一层是构建python3环境+安装依赖包，第二层是在第一层的基础上导入数据程序等文件夹，执行程序。
1.第一层
构建python3环境+安装依赖包
文件结构为：
```buildoutcfg
.
├── Dockerfile
└── code
    └── requirements.txt
```
Dockerfile文件内容
```
# Base Images
## 从第一层镜像构建
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/python:3

## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /

## 安装依赖库
RUN pip install --upgrade pip \ 
 && pip install -r code/requirements.txt

## 恢复到系统命令行模式
CMD ["bash"]
```
构建镜像:在一个新的文件夹下
```bash
xulei$ docker build -t registry.cn-shanghai.aliyuncs.com/xl717/pakdd2020:1.0 .
```

2.第二层
在第一层的基础上导入数据程序等文件夹，执行程序
文件结构为：
```buildoutcfg
.
├── .DS_Store
├── Dockerfile
├── README.md
├── code
│   ├── .DS_Store
│   └── main.py
├── data
│   ├── .DS_Store
│   ├── round2_testA
│   ├── round2_testB
│   │   └── disk_sample_smart_log_test_b.csv.zip
│   └── round2_train
│       ├── .DS_Store
│       ├── disk_sample_fault_tag.csv
│       ├── disk_sample_smart_log_201805.csv
│       └── disk_sample_smart_log_201806.csv
├── feature
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
└── user_data
    ├── .DS_Store
    └── tmp_data
        ├── 201805.csv
        ├── 201806.csv
        └── test_b.csv
```
Dockerfile文件内容
```
# Base Images
## 从天池基础镜像构建
FROM registry.cn-shanghai.aliyuncs.com/xl717/pakdd2020:1.0

## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /

## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /

## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]
```
构建镜像:在一个新的文件夹下
```bash
xulei$ docker build -t registry.cn-shanghai.aliyuncs.com/xl717/pakdd2020:2.0 .
```

本地镜像push到阿里云仓库
```bash
1.登录阿里云docker registry
xulei$ sudo docker login --username=1053428306@qq.com registry.cn-shanghai.aliyuncs.com
2.push
xulei$ sudo docker push registry.cn-shanghai.aliyuncs.com/xl717/pakdd2020:1.0
xulei$ sudo docker push registry.cn-shanghai.aliyuncs.com/xl717/pakdd2020:2.0
```

### 阿里云仓库
[阿里云-pakdd202仓库网址](https://cr.console.aliyun.com/repository/cn-shanghai/xl717/pakdd2020/images?spm=5176.12901015.0.i12901015.6ec2525c6u1eMl)












