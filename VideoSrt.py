from re import split
from select import select
from matplotlib.pyplot import connect
import paddle
from paddlespeech.cli import ASRExecutor
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import os
import threading
import math
from zmq import device
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
def DateString(time,farme=30):
    s=math.floor(time/1000)
    m=math.floor(s/60)
    h=math.floor(m/60)
    f=math.floor((time%1000)/1000*farme)
    s=s%60
    return "{:0>2d}:{:0>2d}:{:0>2d},{:0>3d}".format(h,m,s,f)

srtlist={}
class Window(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        layout = QGridLayout()
        self.setLayout(layout)
        self.resize(300, 400)
        self.setWindowTitle("字幕生成")

        self.def_folder=QLineEdit()
        self.def_folder.setText("V:\\")
        layout.addWidget(self.def_folder) #默认工作文件夹

        self.inputfile=QLineEdit()
        layout.addWidget(self.inputfile) #需要转换的文件

        self.inputfilebutton=QPushButton()
        self.inputfilebutton.setText("1.选择文件")
        layout.addWidget(self.inputfilebutton)
        self.inputfilebutton.clicked.connect(self.selectfile)
        
        self.takewavbutton=QPushButton()
        self.takewavbutton.setText("2.生成wav文件")
        self.takewavbutton.clicked.connect(self.takewav)
        layout.addWidget(self.takewavbutton)

        self.makepartslider1=QSlider()
        self.makepartslider1.setOrientation(Qt.Horizontal)
        self.makepartslider1.setRange(10,1000)
        self.makepartslider1.setValue(50)
        layout.addWidget(self.makepartslider1)

        self.makepartslider2=QSlider()
        self.makepartslider2.setOrientation(Qt.Horizontal)
        self.makepartslider2.setRange(-50,50)
        self.makepartslider2.setValue(-45)
        layout.addWidget(self.makepartslider2)

        self.makepartbutton=QPushButton()
        self.makepartbutton.setText("3.分析音频文件")
        self.makepartlabel=QLabel()
        self.makepartbutton.clicked.connect(self.makepart)
        layout.addWidget(self.makepartbutton)
        layout.addWidget(self.makepartlabel)

        self.makesrtbutton=QPushButton()
        self.makesrtbutton.setText("4.生成字幕")
        self.makesrtbutton.clicked.connect(self.makesrt)
        layout.addWidget(self.makesrtbutton)


    def selectfile(self):
        file,ok=QFileDialog.getOpenFileName(self,"选择文件",".","*.*")
        if file!="":
            self.inputfile.setText(file)
        else:
            print("未选择文件.")
    def takewav(self):
        os.system("ffmpeg.exe -i "+self.inputfile.text()+" -f wav -ar 16000 "+self.def_folder.text()+"out.wav")
    def makepart(self):
        self.audio=AudioSegment.from_wav(self.def_folder.text()+"out.wav")
        #chunks=split_on_silence(audio,100,-45,200)
        try:
            self.chunkstime=detect_nonsilent(self.audio,self.makepartslider1.value(),self.makepartslider2.value(),self.makepartslider1.value())
            chunkstime=self.chunkstime
            time20=0
            time10=0
            time5=0
            time02=0
            print('总分段：', len(chunkstime))
            for i in list(range(len(chunkstime)))[::-1]:
                if chunkstime[i][1]-chunkstime[i][0] <= 100:
                    chunkstime.pop(i)
                    time02+=1
                if chunkstime[i][1]-chunkstime[i][0] >= 20000:
                    time20+=1
                if chunkstime[i][1]-chunkstime[i][0] >= 10000:
                    time10+=1
                if chunkstime[i][1]-chunkstime[i][0] >= 5000:
                    time5+=1
            print('取有效分段(大于0.2s)：', len(chunkstime),time20,time10,time5)
            self.makepartlabel.setText('总分段：{} 有效:{} 大于20s:{} 大于10s:{},大于5s:{} ,小于0.2:{}'.format(len(chunkstime),len(chunkstime),time20,time10,time5,time02))
        except:
            self.makepartlabel.setText('error:数据超出范围')
    def makesrt(self):
        try:
            os.mkdir(self.def_folder.text()+"outwav")
        except:
            print("文件夹已存在")
        chunkstime=self.chunkstime
        audio=self.audio
        use_threads=1 #使用线程数量
        thread_list=[]
        jianju=int(len(chunkstime)/use_threads)
        for i in range(use_threads):
            start=i*jianju
            if i==use_threads-1:
                end=len(chunkstime)
            else:
                end=(i+1)*jianju
            thread_list.append(VoiceToSrt(i,chunkstime[start:end],start,audio,self.def_folder.text()))
        for thread in thread_list:
            thread.start()
        for thread in thread_list:
            thread.join()
            srt=os.open(self.def_folder.text()+"srt.srt",os.O_RDWR|os.O_CREAT)
            for i in range(len(chunkstime)):
                os.write(srt, srtlist[i].encode("utf-8"))
        os.close(srt)
class VoiceToSrt(threading.Thread): 
    def __init__(self, threadID,chunkstime,start,audio,def_folder):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.chunkstime=chunkstime
        self.startid=start
        self.asr_executor = ASRExecutor()
        self.audio=audio
        self.def_folder=def_folder
    def run(self):
        for i, chunk in enumerate(self.chunkstime):
            self.audio[chunk[0]:chunk[1]].export(self.def_folder+"outwav/{}-{}.wav".format(self.threadID,i),format='wav')
            text = self.asr_executor(
            model='conformer_wenetspeech',
            lang='zh',
            sample_rate=16000,
            config=None,  # Set `config` and `ckpt_path` to None to use pretrained model.
            ckpt_path=None,
            audio_file=self.def_folder+'outwav/{}-{}.wav'.format(self.threadID,i),
            force_yes=False,
            device=paddle.get_device())
            print("Device:",paddle.get_device())
            outsrtline="{}\n{} --> {}\n{}\n\n".format(self.startid+i+1,DateString(chunk[0]),DateString(chunk[1]),text)
            srtlist[self.startid+i]=outsrtline
            print("Thread:",self.threadID,text)

#print('ASR Result: \n{}'.format(text))
app = QApplication(sys.argv)
screen = Window()
screen.show()
sys.exit(app.exec_())