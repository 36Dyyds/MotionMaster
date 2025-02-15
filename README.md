# MotionMaster (动作识别大师)

MotionMaster（动作识别大师）是一款基于 Python 的入门级动作识别项目，旨在为用户提供简单易用的图片和视频动作识别功能。无论是初学者还是有一定编程经验的用户，都能够轻松上手，通过该项目快速了解和应用动作识别技术。

## 功能特性

1. 动作识别：支持视频动作识别，后续支持图片、摄像头中的动作识别。
2. 人脸识别：支持图片、视频、摄像头中的人脸识别。可以参考我的另一个项目：FaceMaster
3. 表情识别：包括哀、惊、惧、乐、怒、厌、中七种基本表情。

持续关注本项目后续支持更多实现！！！

## 功能演示

### 动作识别

以下演示视频来源于网络，如有侵权请告知删除！！！

![动作识别](https://gitee.com/qq153128151/MotionMaster/blob/master/output/test/test.jpg)

## 项目环境

- 平台： Windows 10
- 工具：PyCharm 2022.1.2
- Python 版本： 3.8.10

PyCharm最好使用专业版，社区版可能存在部分问题。
TensorFlow 默认不使用 GPU 处理，如需开启请参考`cuda_test.py`请确保 tensorflow 与 Python、CUDA、cuDNN 版本一致。

## 安装依赖

##### 1、克隆本项目

```bash
git clone https://gitee.com/qq153128151/MotionMaster.git
```

或

```bash
git clone https://github.com/36Dyyds/MotionMaster.git
```

##### 2、更换国内镜像

```bash
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set install.trusted-host mirrors.aliyun.com
pip config get global.index-url
```

##### 3、安装依赖

```bash
pip install -r requirements.txt
```

## 运行说明

项目中的代码都有详细的注释，不依赖其他文件右键运行即可。

## 常见问题

- 依赖无法下载：

  使用国内的镜像源，如果开启了代理或科学上网需要关闭。

## 贡献

如果您发现了任何问题或者有任何建议，欢迎提出 issue 或者提交 pull request。

## 群聊

##### 微信交流群

扫码添加微信，备注：MotionMaster或FaceMaster，邀您加入群聊

![微信](https://gitee.com/qq153128151/FaceMaster/raw/master/images/wx.png)

扫码添加微信，备注：MotionMaster或FaceMaster，邀您加入群聊

## 打赏

如果我的项目对你有所帮助还请给个免费的 Star 能让更多人看到。

![微信打赏](https://gitee.com/qq153128151/FaceMaster/raw/master/images/reward.png)

## 版权信息

本项目遵循 MIT License。详细信息请参阅 LICENSE 文件。

希望 MotionMaster 能够帮助您快速入门和应用动作识别技术，祝您使用愉快！