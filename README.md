# 感知增强系统设计

## 安装

* 打开cmd
* 安装相应的库，
  pip install opencv-python numpy os tqdm

## 文件组成

```python
work
├── data #存放待处理的视频文件
├── input # 老师提供的数据
│   │   ├── set1
│   │   ├── set2
├── |   ├── set3
├── output # 处理结果，包含原图和处理之后的结果
│   │   ├── set1
│   │   ├── set2
├── |   ├── set3
├── ui   # 系统界面设计代码
├── config.json # retinex算法的参数
├── dehaze.py   # 去雾算法
├── main.py   # 主函数，图像增强
├── retinex.py # retinex算法
├── visual     # 系统界面
```

## 使用说明

进行图像增强主要有两个文件分别是 main.py 和 visual.py

### **main.py**

通过设置d和e选择去雾还是不均匀光照、低光照的增强，设置输入图片路径和输出路径

```python
d = False # 去雾
e = True # 图像增强
# 图片路径
data_path = 'input/set3'
save_path = 'result/set3'
```

### **visual.py**

利用pyqt5写了增强界面，可进行离线图片的增强、离线视频增强以及实时增强。可实现以下功能：

![1655791020815](https://file+.vscode-resource.vscode-cdn.net/d%3A/python_files4/Retinex-master/image/README_/1655791020815.png)

#### 离线图片增强

选择需要处理的图像，然后点击需要对应的算法，即可完成对离线图片的增强。页面左侧为原始图像，右侧为处理后的图像。

#### 离线视频增强

离线视频增强和图像增强类似，选择对应的视频和算法，即可完成增强，左侧为原始图像，右侧为处理后的视频，同步播放。
需要说明的是在实际的测试过程中，发现Retinex系列算法，运算时间过长，效率较低。因此离线视频增强部分只是选择了
暗通道去雾算法，完成去雾任务。后续还需进一步完善。

#### 实时感知增强

由于硬件条件限制，缺少可用的外设（摄像机），因此为验证该部分功能，将个人手机（iPhone se2）拓展为摄像机进行实验。
具体方法为：在手机端下载安装“IP摄像机Lite软件”，手机和电脑链接在同一局域网下，将手机相机拍摄到的画面实时传输到电脑端完成图像增强，详细内容可参照代码
IP摄像机的使用参考：[使用手机摄像头做网络ip摄像头 并用opencv获取rtsp视频流](https://blog.csdn.net/xiaoqiang_007_/article/details/106578900)

**说明**：默认设置的是打开电脑的摄像头，可以设置相应的地址，利用IP摄像机app传输视频流
可在161行更改。

```python
video="http://admin:admin@192.168.2.34:8081/" #此处@后的ipv4 地址
video_stream =cv2.VideoCapture(video)
```
