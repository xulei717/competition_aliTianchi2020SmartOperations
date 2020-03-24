一、测试镜像能在本地跑通并成功推送到阿里云端
1.构建四个文件：Dockerfile, hello_world.py, result.json, run.sh

2.构建镜像:在四个文件目录下
docker build -t registry.cn-shanghai.aliyuncs.com/test717/tianchi_submit:1.0 .

运行镜像: docker run image的id或镜像名 sh run.sh
docker run registry.cn-shanghai.aliyuncs.com/test717/tianchi_submit:1.0 sh run.sh
运行成功输出一行信息: hello world!!!

如果要手动运行，创建好镜像后，创建一个容器并进入命令行系统运行代码（name你随便起）：

docker run -it --name=c_tianchi 镜像id /bin/bash
输入完成后就进入这个系统里了，下面是linux命令行：
查看文件信息
ls
运行shell脚本
sh run.sh

这样也能进行测试。


运行无误后，登录阿里云：

docker login --username=你的阿里云用户名 registry.cn-shanghai.aliyuncs.com

然后输入密码，即可返回登录成功的信息。

登录成功后可以推送镜像到云端（注意没登陆会被拒绝掉）：

docker push registry.cn-shanghai.aliyuncs.com/test717/tianchi_submit:1.0

这样就推送成功了，第一步完成！


二、project
1.构建镜像：在project目录下
sudo docker build -t registry.cn-shanghai.aliyuncs.com/xl717/pakdd2020:0324.3 .
验证运行镜像
（1）查看目前所有镜像
sudo docker images
（2）执行镜像
sudo docker run registry.cn-shanghai.aliyuncs.com/xl717/pakdd2020:0324.3 sh run.sh
2.

