from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTextEdit, QLabel, 
                             QApplication, QLineEdit, QPushButton, QHBoxLayout,
                             QMainWindow, QFrame, QScrollArea)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QPalette, QColor
import asyncio
import tempfile
import os
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtCore import QUrl

class ChatWindow(QMainWindow):
    """聊天窗口UI"""
    
    # 定義信號
    message_sent = Signal(str)
    
    def __init__(self):
        super().__init__()
        
        # 設置窗口屬性
        self.setWindowTitle("YCM 館長")
        self.setMinimumSize(800, 600)
        
        # 創建主窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 創建主佈局
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # 設置音訊播放器
        self.setup_audio()
        
        # 設置樣式
        self.setup_style()
        
        # 創建UI元素
        self.create_header()
        self.create_chat_area()
        self.create_input_area()
        
        # 連接信號
        self.send_button.clicked.connect(self.send_message)
        self.input_field.returnPressed.connect(self.send_message)
    
    def setup_audio(self):
        """設置音訊播放器"""
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.temp_dir = tempfile.mkdtemp()
        
    def setup_style(self):
        """設置界面樣式"""
        # 設置字體
        self.font = QFont("Microsoft JhengHei", 10)  # 使用微軟正黑體
        self.setFont(self.font)
        
        # 設置調色板
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
        self.setPalette(palette)
        
    def create_header(self):
        """創建頂部標題區域"""
        header_frame = QFrame()
        header_frame.setFrameShape(QFrame.StyledPanel)
        header_frame.setStyleSheet("background-color: #0078D7; color: white; border-radius: 5px;")
        
        header_layout = QHBoxLayout(header_frame)
        
        # 標題
        title_label = QLabel("YCM 館長")
        title_label.setFont(QFont("Microsoft JhengHei", 16, QFont.Bold))
        header_layout.addWidget(title_label)
        
        # 狀態
        self.status_label = QLabel("已連接")
        header_layout.addWidget(self.status_label, alignment=Qt.AlignRight)
        
        self.main_layout.addWidget(header_frame)
        
    def create_chat_area(self):
        """創建聊天記錄顯示區域"""
        # 滾動區域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # 聊天記錄容器
        chat_container = QWidget()
        self.chat_layout = QVBoxLayout(chat_container)
        self.chat_layout.setAlignment(Qt.AlignTop)
        
        # 預設間距
        self.chat_layout.setContentsMargins(10, 10, 10, 10)
        self.chat_layout.setSpacing(10)
        
        # 設置滾動區域的內容
        scroll_area.setWidget(chat_container)
        
        # 添加到主佈局
        self.main_layout.addWidget(scroll_area, stretch=1)
        
    def create_input_area(self):
        """創建底部輸入區域"""
        input_frame = QFrame()
        input_frame.setFrameShape(QFrame.StyledPanel)
        input_frame.setStyleSheet("background-color: #F0F0F0; border-radius: 5px;")
        
        input_layout = QHBoxLayout(input_frame)
        
        # 輸入框
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("請輸入消息...")
        self.input_field.setStyleSheet("background-color: white; border: 1px solid #CCCCCC; border-radius: 3px; padding: 5px;")
        input_layout.addWidget(self.input_field)
        
        # 發送按鈕
        self.send_button = QPushButton("發送")
        self.send_button.setStyleSheet("background-color: #0078D7; color: white; border-radius: 3px; padding: 5px 15px;")
        input_layout.addWidget(self.send_button)
        
        # 添加到主佈局
        self.main_layout.addWidget(input_frame)
        
    def send_message(self):
        """發送消息"""
        message = self.input_field.text().strip()
        if message:
            # 顯示用戶消息
            self.add_message_bubble(message, is_user=True)
            
            # 清空輸入框
            self.input_field.clear()
            
            # 發出信號
            self.message_sent.emit(message)
            
    def add_message_bubble(self, message, is_user=False):
        """添加消息氣泡
        
        Args:
            message: 消息內容
            is_user: 是否為用戶消息
        """
        bubble_frame = QFrame()
        bubble_frame.setFrameShape(QFrame.StyledPanel)
        
        # 設置樣式
        if is_user:
            bubble_frame.setStyleSheet("background-color: #DCF8C6; border-radius: 10px; padding: 10px;")
        else:
            bubble_frame.setStyleSheet("background-color: white; border-radius: 10px; padding: 10px;")
        
        bubble_layout = QVBoxLayout(bubble_frame)
        bubble_layout.setContentsMargins(10, 10, 10, 10)
        
        # 消息文本
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bubble_layout.addWidget(message_label)
        
        # 添加到聊天布局
        message_container = QHBoxLayout()
        message_container.setContentsMargins(0, 0, 0, 0)
        
        if is_user:
            # 用戶消息靠右
            message_container.addStretch()
            message_container.addWidget(bubble_frame, alignment=Qt.AlignRight)
        else:
            # 助手消息靠左
            message_container.addWidget(bubble_frame, alignment=Qt.AlignLeft)
            message_container.addStretch()
        
        self.chat_layout.addLayout(message_container)
        
        # 滾動到底部
        self.scroll_to_bottom()
        
    def scroll_to_bottom(self):
        """滾動到底部"""
        # 尋找滾動區域
        for child in self.central_widget.findChildren(QScrollArea):
            # 滾動到底部
            scrollbar = child.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            break
        
    def show_message(self, message):
        """顯示助手消息
        
        Args:
            message: 消息內容
        """
        self.add_message_bubble(message, is_user=False)
        
        # 暫時禁用 Google TTS，因為我們已經在 main.py 中使用 ElevenLabs TTS
        # asyncio.run(self.text_to_speech(message))
        
    def set_status(self, status):
        """設置狀態
        
        Args:
            status: 狀態文本
        """
        self.status_label.setText(status)
