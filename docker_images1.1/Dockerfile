# Base Images
## 从天池基础镜像构建
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/python:3

## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /

## 安装依赖库
RUN pip install --upgrade pip \ 
 && pip install -r code/requirements.txt

## 恢复到系统命令行模式
CMD ["bash"]