# Base Images
## 从天池基础镜像构建
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/python:3

## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /

## 更新ubuntu下载源
ADD sources.list /etc/apt/
RUN apt-get update \
 && apt-get upgrade

## 更改pip源
# ADD pip.conf ~/.pip/pip.conf

## 更新pip到最新
# RUN pip install --upgrade pip

## 配置阿里源
#RUN pip config set global.index-url http://mirrors.aliyun.com/pypi/simple \
# && pip config set install.trusted-host mirrors.aliyun.com

## 安装依赖库
# RUN pip install -r code/requirements.txt
RUN pip install --upgrade pip \
 && pip install  -r code/requirements.txt -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com

## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /

## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]
