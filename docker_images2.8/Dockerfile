# Base Images
## 从天池基础镜像构建
FROM registry.cn-shanghai.aliyuncs.com/xl717/pakdd2020:1.1

# 安装vim
# RUN apt-get update \
#     && apt-get install vim

## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /

## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /

## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]