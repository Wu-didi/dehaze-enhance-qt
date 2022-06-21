from pickle import TRUE
import sys


from PyQt5.QtWidgets import QFileDialog,QTableWidgetItem,QMainWindow,QApplication,QMessageBox,QPushButton
from PyQt5.QtCore import QTimer
from PyQt5 import QtGui
from PyQt5.QtGui import *
from defusedxml import DTDForbidden
from ui.ui import Ui_MainWindow
import retinex
from dehaze import dehaze
from json import load
import cv2
import numpy as np

with open('config.json', 'r') as f:
    config = load(f)



class myclass(QMainWindow,Ui_MainWindow):
    def __init__(self,parent=None):
        super(myclass, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.openImage)
        self.pushButton_2.clicked.connect(self.deha) #去雾
        self.pushButton_4.clicked.connect(self.retinex2)#retinex2
        self.pushButton_7.clicked.connect(self.retinex3)  # retinex3
        self.pushButton_3.clicked.connect(self.retinex1) #retinex
        self.pushButton_5.clicked.connect(self.on_carme)  # 摄像头
        self.pushButton_6.clicked.connect(self.on_video)  # 视频检测
        self.label.setScaledContents(True)#设置图片尺寸自适应
        self.label_2.setScaledContents(True)#设置图片尺寸自适应
        self.label_3.setScaledContents(True)  # 设置图片尺寸自适应
        self.open_flag = False
        self.painter = QPainter(self)#QPainter对小部件和其他绘图设备执行低级绘制。它可以绘制从简单线条到复杂形状的所有内容。

    
    def openImage(self):
        
        self.label.clear()#设置图片尺寸自适应
        self.label_2.clear()#设置图片尺寸自适应
        # 打开一张图片
        # 通过getOpenFileName打开对话框获取一张图片
        try:
            self.path, ret = QFileDialog.getOpenFileName(self, "打开图片", r"./input/",)
            # 把图片转换成BASE64编码
            self.label.setPixmap(QPixmap(self.path))
            self.textEdit.setText(self.path)
        except:
            self.textEdit.setText("'Open Error! Try again!'")

    
    def retinex1(self):
        img = cv2.imread(self.path) 
        img_msrcr = retinex.MSRCR(
            img,
            config['sigma_list'],
            config['G'],
            config['b'],
            config['alpha'],
            config['beta'],
            config['low_clip'],
            config['high_clip']
        )
        image_height, image_width, image_depth = img_msrcr.shape  # 获取图像的高，宽以及深度。

        QIm = cv2.cvtColor(img_msrcr, cv2.COLOR_BGR2RGB)  # opencv读图片是BGR，qt显示要RGB，所以需要转换一下
        QIm = QImage(QIm.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                    image_width * image_depth,
                    QImage.Format_RGB888)
        self.label_2.setPixmap(QPixmap.fromImage(QIm))

    def deha(self):
        img = cv2.imread(self.path)
        img_msrcr = dehaze(
            img
        )
        image_height, image_width, image_depth = img_msrcr.shape  # 获取图像的高，宽以及深度。


        img_msrcr=img_msrcr.astype(np.uint8)  #python类型转换


        QIm = cv2.cvtColor(img_msrcr, cv2.COLOR_BGR2RGB)  # opencv读图片是BGR，qt显示要RGB，所以需要转换一下
        QIm = QImage(QIm.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                    image_width * image_depth,
                    QImage.Format_RGB888)
        self.label_2.setPixmap(QPixmap.fromImage(QIm))

    def retinex2(self):
        img = cv2.imread(self.path)
        img_msrcr = retinex.automatedMSRCR(
                        img,
                        config['sigma_list']
      
        )
        image_height, image_width, image_depth = img_msrcr.shape  # 获取图像的高，宽以及深度。

        QIm = cv2.cvtColor(img_msrcr, cv2.COLOR_BGR2RGB)  # opencv读图片是BGR，qt显示要RGB，所以需要转换一下
        QIm = QImage(QIm.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                    image_width * image_depth,
                    QImage.Format_RGB888)
        self.label_2.setPixmap(QPixmap.fromImage(QIm))
    
    def retinex3(self):
        img = cv2.imread(self.path)
        img_msrcr = retinex.MSRCP(
            img,
            config['sigma_list'],
            config['low_clip'],
            config['high_clip']  
        )
        image_height, image_width, image_depth = img_msrcr.shape  # 获取图像的高，宽以及深度。

        QIm = cv2.cvtColor(img_msrcr, cv2.COLOR_BGR2RGB)  # opencv读图片是BGR，qt显示要RGB，所以需要转换一下
        QIm = QImage(QIm.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                    image_width * image_depth,
                    QImage.Format_RGB888)
        self.label_2.setPixmap(QPixmap.fromImage(QIm))

    def on_video(self):
        self.label.clear()#设置图片尺寸自适应
        self.label_2.clear()#设置图片尺寸自适应
        self.video_stream = cv2.VideoCapture(r'./data/cross.avi')
        if self.open_flag:
            self.pushButton_6.setText('视频检测-open')
        else:
            self.pushButton_6.setText('视频检测-close')
        self.open_flag = bool(1 - self.open_flag)  #


    def paintEvent(self, a0: QPaintEvent):
        if self.open_flag:
            ret, frame = self.video_stream.read()
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # cv2.imshow('test',frame)
            # cv2.waitKey(10)
            self.Qframe = QImage(frame.data, frame.shape[1], frame.shape[0], 
                                    frame.shape[1] * 3, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(self.Qframe))

            img_msrcr = dehaze(frame)
            image_height, image_width, image_depth = img_msrcr.shape  # 获取图像的高，宽以及深度。

            img_msrcr=img_msrcr.astype(np.uint8)  #python类型转换
            img_msrcr = cv2.cvtColor(img_msrcr, cv2.COLOR_BGR2RGB)  # opencv读图片是BGR，qt显示要RGB，所以需要转换一下
            QIm = QImage(img_msrcr.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                        image_width * image_depth,
                        QImage.Format_RGB888)
            self.label_2.setPixmap(QPixmap.fromImage(QIm))
            self.update()


    def on_carme(self):
        self.label.clear()#设置图片尺寸自适应
        self.label_2.clear()#设置图片尺寸自适应

        # video="http://admin:admin@192.168.2.34:8081/" #此处@后的ipv4 地址需要修改为自己的地址
        video = 0
        # 参数为0表示打开内置摄像头，参数是视频文件路径则打开视频
        self.video_stream =cv2.VideoCapture(video)

        if self.open_flag:
            self.pushButton_5.setText('摄像机-open')
        else:
            self.pushButton_5.setText('摄像机-close')
        self.open_flag = bool(1 - self.open_flag)  #
    
    # 未使用
    def detect_carme(self):

        video = 0
        # 参数为0表示打开内置摄像头，参数是视频文件路径则打开视频
        self.video_stream =cv2.VideoCapture(video)
        ret =  1
        while ret:
            ret, frame = self.video_stream.read()
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #cv2.imshow('test',frame)
            cv2.waitKey(10)
            self.Qframe = QImage(frame.data, frame.shape[1], frame.shape[0], 
                                    frame.shape[1] * 3, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(self.Qframe))

            img_msrcr = dehaze(
                frame,
     
        )
            image_height, image_width, image_depth = img_msrcr.shape  # 获取图像的高，宽以及深度。

            img_msrcr=img_msrcr.astype(np.uint8)  #python类型转换
            img_msrcr = cv2.cvtColor(img_msrcr, cv2.COLOR_BGR2RGB)  # opencv读图片是BGR，qt显示要RGB，所以需要转换一下
            QIm = QImage(img_msrcr.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                        image_width * image_depth,
                        QImage.Format_RGB888)
            self.label_2.setPixmap(QPixmap.fromImage(QIm))
            self.update()


    def closeEvent(self, event):
        ok = QPushButton()
        cancel =QPushButton()
        msg = QMessageBox(QMessageBox.Warning, u'关闭', u'是否关闭！')
        msg.addButton(ok, QMessageBox.ActionRole)
        msg.addButton(cancel,QMessageBox.RejectRole)
        ok.setText(u'确定')
        cancel.setText(u'取消')
        if msg.exec_() == QMessageBox.RejectRole:
            event.ignore()
        else:
            # if self.cap.isOpened():
            #     self.cap.release()
            # if self.timer_camera.isActive():
            #     self.timer_camera.stop()
            event.accept()

    
#  路径中包含中文识别不出来  要被显示的图片路径（Resources/小狗.jpg）中含有中文。这个时候我们可以使用使用QString的fromLocal8Bit()函数，实现从本地字符集GBK到Unicode的转换。我们将Label.cpp的代码改成如下所示：
if __name__ == '__main__':
    app=QApplication(sys.argv)
    ui = myclass()
    ui.show()
    sys.exit(app.exec_())

    
