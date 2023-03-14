import os
import sys
from PyQt5 import QtGui, QtWidgets, QtCore


class UiMainWindow(QtWidgets.QWidget):
    def __init__(self):
        super(UiMainWindow, self).__init__()
        self.setObjectName("MainWindow")
        self.setWindowTitle("自动标注")
        # self.setFixedSize(1600, 740)
        self.resize(1600, 720)

        self.font = QtGui.QFont()
        self.font.setFamily("宋体")
        self.font.setPointSize(14)

        self.gridLayout = QtWidgets.QGridLayout()
        self.setLayout(self.gridLayout)

        self.label_image = QtWidgets.QLabel()  # 图片
        self.viceWidget = QtWidgets.QWidget()
        self.viceLayout = QtWidgets.QGridLayout()
        self.viceWidget.setLayout(self.viceLayout)

        self.button_label = QtWidgets.QPushButton("自动标注")  # 标注按键
        self.button_filter = QtWidgets.QPushButton("图片筛选")  # 图片筛选按键
        self.button_select = QtWidgets.QPushButton("选择标注视频")  # 文件选择按键
        self.label_file = QtWidgets.QLineEdit("")  # 文件路径
        self.label_fps = QtWidgets.QLineEdit("帧数：")  # 总帧数，图片个数
        self.button_auto = QtWidgets.QPushButton("开启自动")  # 剔除键
        self.button_last = QtWidgets.QPushButton("上一帧")  # 确认
        self.button_clear = QtWidgets.QPushButton("剔除")  # 剔除键
        self.button_yes = QtWidgets.QPushButton("确认")  # 确认
        self.timer_video = QtCore.QTimer()  # 初始化定时器

        self.setup_ui()
        self.set_connect()

        self.isvideo = True
        self.auto_label = False
        self.path = ""
        self.root_video_dir = "E:/MachineVision/DataSources/BeltConveyor/SplitVideo"
        self.root_image_dir = "E:/MachineVision/DataSources/BeltConveyor/ImageAndLabel"
        self.save_img_dir = ''
        self.save_txt_dir = ''

    def setup_ui(self):
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout.setSpacing(10)

        self.gridLayout.addWidget(self.label_image, 0, 0, 1, 4)
        self.label_image.setStyleSheet("background-color: white")

        self.gridLayout.addWidget(self.viceWidget, 0, 4, 1, 1)

        self.viceLayout.addWidget(self.button_label, 0, 0, 1, 1)
        self.button_label.setFont(self.font)

        self.viceLayout.addWidget(self.button_filter, 0, 1, 1, 1)
        self.button_filter.setFont(self.font)

        self.viceLayout.addWidget(self.button_select, 1, 0, 1, 2)
        self.button_select.setFont(self.font)

        self.viceLayout.addWidget(self.label_file, 2, 0, 1, 2)
        self.label_file.setFont(self.font)
        self.label_file.setReadOnly(True)
        self.label_file.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.viceLayout.addWidget(self.label_fps, 3, 0, 1, 2)
        self.label_fps.setFont(self.font)
        self.label_fps.setReadOnly(True)
        self.label_fps.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        self.viceLayout.addWidget(self.button_auto, 4, 0, 1, 1)
        self.button_auto.setFont(self.font)

        self.viceLayout.addWidget(self.button_last, 4, 1, 1, 1)
        self.button_last.setFont(self.font)

        self.viceLayout.addWidget(self.button_clear, 5, 0, 1, 1)
        self.button_clear.setFont(self.font)

        self.viceLayout.addWidget(self.button_yes, 5, 1, 1, 1)
        self.button_yes.setFont(self.font)

    def set_connect(self):
        self.button_label.clicked.connect(self.set_label)
        self.button_filter.clicked.connect(self.set_filter)
        self.button_select.clicked.connect(self.select_file)
        self.button_yes.clicked.connect(self.set_next)
        self.button_last.clicked.connect(self.set_last)
        self.button_clear.clicked.connect(self.set_remove)
        self.button_auto.clicked.connect(self.auto_video)
        self.timer_video.timeout.connect(self.next_video)

    def set_label(self):
        self.isvideo = True
        self.button_select.setText("选择标注视频")
        self.label_file.clear()
        self.label_fps.setText("帧数：")
        self.path = ""

    def set_filter(self):
        self.isvideo = False
        self.button_select.setText("选择筛选文件夹")
        self.label_file.clear()
        self.label_fps.setText("帧数：")
        self.path = ""

    def set_next(self):
        if self.isvideo:
            self.next_video()
        else:
            self.next_image()

    def set_last(self):
        if self.isvideo:
            self.last_video()
        else:
            self.last_image()

    def set_remove(self):
        if self.isvideo:
            self.nosave_video()
        else:
            self.remove_image()

    def select_file(self):
        if self.isvideo:
            path = QtWidgets.QFileDialog.getOpenFileName(
                self, "选取视频", self.root_video_dir, '视频: *.avi, *.mp4')[0]
            if path:
                self.path = path
                video_name = os.path.basename(path).split('.')[0][13:].lower()
                video_type = os.path.basename(os.path.dirname(path))
                self.save_img_dir = os.path.join(self.root_image_dir, video_type, video_name, 'images')
                self.save_txt_dir = os.path.join(self.root_image_dir, video_type, video_name, 'labels')
                if not os.path.exists(os.path.join(self.root_image_dir, video_type)):
                    os.mkdir(os.path.join(self.root_image_dir, video_type))
                if not os.path.exists(os.path.join(self.root_image_dir, video_type, video_name)):
                    os.mkdir(os.path.join(self.root_image_dir, video_type, video_name))
                if not os.path.exists(self.save_img_dir):
                    os.mkdir(self.save_img_dir)
                    os.mkdir(self.save_txt_dir)
                self.label_file.setText(self.path)
                self.open_picture()
        else:
            path = QtWidgets.QFileDialog.getExistingDirectory(
                self, "选取文件夹", self.root_image_dir)  # 读取文件夹
            if path and ('labels' in os.listdir(path) and 'images' in os.listdir(path)):
                self.path = path
                self.label_file.setText(self.path)
                self.open_picture()
            else:
                self.label_file.setText("请选择正确文件夹")

    def open_picture(self):
        pass

    def auto_video(self):
        pass

    def next_image(self):
        pass

    def next_video(self):
        pass

    def last_image(self):
        pass

    def last_video(self):
        pass

    def remove_image(self):
        pass

    def nosave_video(self):
        pass


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = UiMainWindow()
    ui.show()
    sys.exit(app.exec_())
