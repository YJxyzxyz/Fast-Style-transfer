import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from evaluate import for_UI


# 调整图片大小
def shrinkImage(img_path, output_height):
    '''
    缩小图片
    :return:
    '''
    # 固定高度
    # scale = 0.8     #每次缩小20%
    img = QImage(img_path)  # 创建图片实例
    # print(img.width())
    scale = output_height / img.height()  # 缩放比例
    output_width = int(img.width() * scale)
    size = QSize(output_width, output_height)
    pixImg = QPixmap.fromImage(img.scaled(size, Qt.IgnoreAspectRatio))
    return pixImg


# 先是组件，组件完事之后再添加布局
class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('基于Fast StyleTransfer图像风格迁移系统')
        self.setWindowIcon(QIcon('images/logo.png'))
        self.resize(1600, 600)
        self.initUI()

    def initUI(self):
        # 主布局
        # addWidget可以添加方式，比如第一个位置是父窗口，第二个参数为间距，第三个参数为方式
        # QFont(字体样式，大小，加粗QFont.bold)
        main_widget = QWidget()
        about_widget = QWidget()
        generally_font = QFont('楷体', 20)
        main_layout = QHBoxLayout()

        # 左边窗口
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        label_input = QLabel("原图")
        # label_input.setAlignment(Qt.AlignCenter)
        label_input.setFont(generally_font)
        self.img_input = QLabel()
        self.input_image_path = "images/oysc.jpg"
        self.img_input.setPixmap(QPixmap(shrinkImage(self.input_image_path, 300)))
        upload_btn = QPushButton(" 选择图片 ")
        upload_btn.setFont(generally_font)
        upload_btn.clicked.connect(self.chose_file)
        self.cb = QComboBox()
        self.cb.addItem("la_muse")
        self.cb.addItem("rain_princess")
        self.cb.addItem("scream")
        self.cb.addItem("udnie")
        self.cb.addItem("wave")
        self.cb.addItem("wreck")
        self.cb.setFont(generally_font)

        left_layout.addWidget(label_input, 0, Qt.AlignCenter | Qt.AlignTop)
        left_layout.addWidget(self.img_input, 0, Qt.AlignCenter)
        left_layout.addWidget(upload_btn, 0, Qt.AlignCenter)
        left_layout.addWidget(self.cb, 0, Qt.AlignCenter)
        left_widget.setLayout(left_layout)

        # 中间窗口
        middle_widget = QWidget()
        middle_layout = QVBoxLayout()
        label_title_2 = QLabel('迁移结果')
        label_title_2.setFont(generally_font)
        self.img_middle = QLabel("中间结果")
        # img_middle = QLabel()
        self.img_middle.setPixmap(QPixmap(shrinkImage('images/temp_transfer.jpg', 300)))
        btn_chong = QPushButton(" 风格迁移 ")
        btn_chong.setFont(generally_font)
        btn_chong.clicked.connect(self.change_style)
        xx = QLabel()
        xx.setFont(generally_font)

        middle_layout.addWidget(label_title_2, 0, Qt.AlignCenter | Qt.AlignTop)
        middle_layout.addWidget(self.img_middle, 0, Qt.AlignCenter)
        middle_layout.addWidget(xx, 0, Qt.AlignCenter)
        middle_layout.addWidget(btn_chong, 0, Qt.AlignCenter)
        middle_widget.setLayout(middle_layout)

        # 关于界面
        about_layout = QVBoxLayout()
        about_title = QLabel('欢迎使用风格迁移系统\n'
                             'QQ:718005487')
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('images/logo.png'))
        about_img.setAlignment(Qt.AlignCenter)
        label_super = QLabel()
        label_super.setText("<a href='http://yjxyzxyz.cn'>我的个人主页</a>")
        label_super.setFont(QFont('楷体', 12))
        label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        # git_img = QMovie('images/')
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)

        # 主页面设置
        main_layout.addWidget(left_widget)
        # main_layout.addStretch(0)
        main_layout.addWidget(middle_widget)
        # main_layout.addWidget(right_widget)
        main_widget.setLayout(main_layout)
        self.addTab(main_widget, '主页面')
        self.addTab(about_widget, '关于')
        self.setTabIcon(0, QIcon('images/主页面.png'))
        self.setTabIcon(1, QIcon('images/关于.png'))

    def chose_file(self):
        print("选择图片")
        fname, _ = QFileDialog.getOpenFileNames(self, 'oepn file',
                                                'D:\\ubuntu\\transfer\\fast-style-transfer_lff\\images',
                                                "Image files(*.jpg *png)")
        # print(fname)
        if len(fname) > 0:
            self.input_image_path = fname[0]
            self.img_input.setPixmap(QPixmap(shrinkImage(self.input_image_path, 300)))
        # 改变图片

    def change_style(self):
        print("风格迁移")
        inpath = self.input_image_path
        checkdir = 'models/' + self.cb.currentText() + '.ckpt'
        out_path = 'images/temp_transfer.jpg'
        for_UI(inpath, out_path, checkdir)
        self.img_middle.setPixmap(QPixmap(shrinkImage('images/temp_transfer.jpg', 300)))

    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     'quit',
                                     "Are you sure?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
