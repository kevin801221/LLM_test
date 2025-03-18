#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
import argparse
import threading
import time
from datetime import datetime
from queue import Queue
from dotenv import load_dotenv

# UI Components
from PySide6.QtWidgets import QApplication
from gui.chat_window import ChatWindow

# Flask 服務
from flask import Flask
from services.common_service import common_bp, rtsp_event_queue

# Utilities
from utils.resource_monitor import ResourceMonitor
from utils.speech_input import SpeechRecognizer
from utils.conversation_memory import EnhancedConversationMemory
from utils.meeting_recorder import MeetingRecorderIntegration

# Services
from services.face_service import FaceService
from services.api_client import get_employee_data
from services.llm_service import LLMService

# 導入 ElevenLabs TTS 服務
from utils.elevenlabs_tts import ElevenLabsTTS

# 導入 RTSP 管理器
from utils.rtsp_manager import RTSPManager, process_rtsp_events

# 導入 pygame 用於音頻播放
import pygame

# Configuration
from config import CAMERA_WIDTH, CAMERA_HEIGHT, parse_arguments
from utils.conversation.mongodb_memory_agent import response_generate, initialize_mongodb_collections
from utils.conversation.mongodb_config import get_mongo_client
from utils.conversation.db_init import DB_NAME

# 初始化全局變數
mongo_client = None
mongo_db = None

def initialize_mongodb(args):
    """初始化 MongoDB 連接和集合"""
    global mongo_client, mongo_db
    
    print("正在初始化 MongoDB 連接...")
    try:
        # 檢查環境變數
        if not os.environ.get("MONGO_URI"):
            # 如果沒有設置 MONGO_URI，使用本地 MongoDB
            os.environ["MONGO_URI"] = "mongodb://localhost:27017/"
            print("未設置 MONGO_URI 環境變數，使用默認本地連接: mongodb://localhost:27017/")
        
        mongo_client = get_mongo_client()
        if mongo_client is not None:
            mongo_db = mongo_client[DB_NAME]
            print("MongoDB 連接成功")
            
            # 根據命令行參數選擇性地清空集合
            print("正在初始化 MongoDB 集合...")
            if hasattr(args, 'init_chat_history') and args.init_chat_history:
                print("根據參數清空聊天歷史集合...")
                initialize_mongodb_collections(force_reinit=False, init_chat_history=True)
            elif hasattr(args, 'init_memory_stream') and args.init_memory_stream:
                print("根據參數清空記憶流集合...")
                initialize_mongodb_collections(force_reinit=False, init_memory_stream=True)
            else:
                initialize_mongodb_collections(force_reinit=False)
            
            print("MongoDB 集合初始化完成")
            return True
        else:
            print("無法獲取 MongoDB 客戶端，請確保 MongoDB 服務正在運行")
    except Exception as e:
        print(f"MongoDB 初始化失敗: {str(e)}")
        mongo_client = None
        mongo_db = None
    
    return False

def create_flask_app():
    app = Flask(__name__)
    app.register_blueprint(common_bp)
    return app

def run_flask():
    app = create_flask_app()
    app.run(host='0.0.0.0', port=5000)

def main():
    """Main application entry point"""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    args = parse_arguments()

    # 初始化 MongoDB
    mongodb_initialized = initialize_mongodb(args)
    if not mongodb_initialized:
        print("警告: MongoDB 未初始化，某些功能可能無法正常工作")
        
    # 初始化本地攝像頭或 RTSP 管理器
    local_cap = None
    rtsp_manager = None
    
    if args.no_rtsp:
        # 使用本地攝像頭
        print("使用本地攝像頭模式")
        # 嘗試多個可能的攝像頭設備
        for camera_id in range(2):  # 嘗試前兩個攝像頭設備
            print(f"嘗試開啟攝像頭 {camera_id}...")
            local_cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)  # 在 Windows 上使用 DirectShow
            if local_cap.isOpened():
                # 檢查是否能實際讀取幀
                ret, test_frame = local_cap.read()
                if ret and test_frame is not None:
                    print(f"成功開啟攝像頭 {camera_id}")
                    break
                else:
                    print(f"攝像頭 {camera_id} 無法讀取影像")
                    local_cap.release()
                    local_cap = None
            else:
                print(f"無法開啟攝像頭 {camera_id}")
        
        if local_cap is None:
            print("錯誤：無法開啟任何本地攝像頭")
            print("請檢查：")
            print("1. 攝像頭是否正確連接")
            print("2. 攝像頭驅動是否正確安裝")
            print("3. 其他程式是否正在使用攝像頭")
            print("4. Windows 設定中的攝像頭權限是否開啟")
            return
    else:
        # 使用 RTSP 管理器
        print("使用 RTSP 串流模式")
        rtsp_manager = RTSPManager()
        
        # 啟動 RTSP 事件處理線程
        rtsp_event_thread = threading.Thread(
            target=process_rtsp_events,
            args=(rtsp_manager,),
            daemon=True
        )
        rtsp_event_thread.start()
        
        # 啟動 Flask 服務
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()

    
    # Initialize Qt application
    qt_app = QApplication.instance()
    if not qt_app:
        qt_app = QApplication([])
    
    # Create chat window
    chat_window = ChatWindow()
    chat_window.show()
    
    # 先顯示測試消息
    chat_window.show_message("系統啟動測試：請問您能看到這條消息嗎？")

    # 檢查 API 密鑰
    if args.model in ["gpt4o", "gpt-4o", "gpt-4"]:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("警告：找不到 OPENAI_API_KEY 環境變數！")
            chat_window.show_message("警告: OpenAI API 密鑰未設置，系統可能無法正常工作。")
    
    chat_window.show_message(f"系統啟動: 使用 {args.model} 模型，如果您看到此消息，界面正常工作。")
    
    # Initialize resource monitor
    resource_monitor = ResourceMonitor(target_cpu_percent=args.cpu_limit)
    resource_monitor.start_monitoring()
    
    # Initialize conversation memory
    conversation_memory = EnhancedConversationMemory()
    
    # Initialize LLM service
    llm_service = LLMService(model_name=args.model)
    
    # Initialize face service
    face_service = FaceService()
    
    # Initialize speech recognition if voice is enabled
    speech_recognizer = None
    if args.use_voice:
        speech_recognizer = SpeechRecognizer()
    
    # Initialize TTS services
    elevenlabs_tts = None
    inai_tts = None
    chatgpt_tts = None
    if args.use_voice:
        try:
            # 優先嘗試初始化 ChatGPT TTS 服務
            from services.chatgpt_tts_service import ChatGPTTTSService
            chatgpt_tts = ChatGPTTTSService()
            if not chatgpt_tts.initialized:
                chatgpt_tts = None
                print("ChatGPT TTS 服務初始化失敗，嘗試使用其他 TTS 服務")
            else:
                elevenlabs_tts = None
                inai_tts = None
                print("使用 ChatGPT TTS 服務")
        except Exception as e:
            print(f"初始化 ChatGPT TTS 服務時出錯: {e}")
            chatgpt_tts = None
            
            # 如果 ChatGPT TTS 初始化失敗，嘗試使用 ElevenLabs TTS
            try:
                # 嘗試初始化 ElevenLabs TTS 服務
                from utils.elevenlabs_tts import ElevenLabsTTS
                elevenlabs_tts = ElevenLabsTTS()
                if not elevenlabs_tts.initialized:
                    elevenlabs_tts = None
                    print("ElevenLabs TTS 服務初始化失敗，嘗試使用 Inai TTS 服務")
                else:
                    inai_tts = None
                    print("使用 ElevenLabs TTS 服務")
            except Exception as e:
                print(f"初始化 ElevenLabs TTS 時出錯: {e}")
                elevenlabs_tts = None
                
            # 如果 ElevenLabs TTS 初始化失敗，嘗試使用 Inai TTS
            if elevenlabs_tts is None:
                try:
                    from services.inai_tts_service import InaiTTSService
                    inai_tts = InaiTTSService()
                    if not inai_tts.initialized:
                        inai_tts = None
                        print("警告: Inai TTS 初始化失敗。語音功能將不可用。")
                except Exception as e:
                    print(f"初始化 Inai TTS 服務時出錯: {e}")
                    inai_tts = None
                else:
                    print("使用 Inai TTS 服務")
    else:
        elevenlabs_tts = None
        chatgpt_tts = None
        inai_tts = None

    # Initialize LangGraph conversation
    langgraph_conversation = None
    try:
        from services.langgraph_service import LangGraphConversation
        langgraph_conversation = LangGraphConversation()
    except Exception as e:
        print(f"無法初始化 LangGraph 對話服務: {e}")
    
    # Initialize queues for feature matching
    feature_queue = Queue()
    result_queue = Queue()
    
    # Initialize meeting recorder
    meeting_recorder = MeetingRecorderIntegration()
    
    # 播放 YCM 公司介紹
    play_ycm_introduction(elevenlabs_tts, inai_tts, chatgpt_tts)
    
    # Start face recognition
    realtime_face_recognition(
        args=args,
        chat_window=chat_window,
        resource_monitor=resource_monitor,
        conversation_memory=conversation_memory,
        llm_service=llm_service,
        face_service=face_service,
        speech_recognizer=speech_recognizer,
        elevenlabs_tts=elevenlabs_tts,
        inai_tts=inai_tts,
        chatgpt_tts=chatgpt_tts,
        langgraph_conversation=langgraph_conversation,
        feature_queue=feature_queue,
        result_queue=result_queue,
        rtsp_manager=rtsp_manager,
        local_cap=local_cap,
        meeting_recorder=meeting_recorder
    )
    
    # Start Qt event loop
    qt_app.exec()

def play_ycm_introduction(elevenlabs_tts=None, inai_tts=None, chatgpt_tts=None):
    """在 Emma 啟動時播放 YCM 公司介紹"""
    try:
        # 優先播放一分鐘摘要版本
        one_minute_intro_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                          "new_data", 
                                          "task_1741932633_可以幫我寫一篇關於YCM優克美公司對黴菌的研究的報告書_chinese_one_minute_summary.mp3")
        
        intro_mp3_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     "new_data", 
                                     "task_1741932633_可以幫我寫一篇關於YCM優克美公司對黴菌的研究的報告書_chinese.mp3")
        
        intro_txt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     "new_data", 
                                     "task_1741932633_可以幫我寫一篇關於YCM優克美公司對黴菌的研究的報告書_chinese.txt")
        
        # 優先播放一分鐘摘要版本
        if os.path.exists(one_minute_intro_path):
            print("播放 YCM 公司一分鐘精彩介紹...")
            
            # 初始化 pygame 混音器
            pygame.mixer.init()
            
            # 加載並播放音頻
            pygame.mixer.music.load(one_minute_intro_path)
            pygame.mixer.music.play()
            
            # 等待音頻播放完成
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # 關閉混音器
            pygame.mixer.quit()
            print("YCM 公司一分鐘精彩介紹播放完成")
            return
        
        # 如果一分鐘摘要不存在，嘗試使用 TTS 朗讀 TXT 文件
        if os.path.exists(intro_txt_path):
            print("播放 YCM 公司介紹...")
            
            try:
                with open(intro_txt_path, 'r', encoding='utf-8') as f:
                    intro_text = f.read()
                
                # 使用前三段文本，包括完整的摘要段落和第三段的前兩句
                paragraphs = intro_text.split('\n\n')
                if len(paragraphs) >= 3:
                    # 標題 + 完整摘要 + 第三段的前兩句
                    short_intro = paragraphs[0] + '\n\n' + paragraphs[1]
                    if len(paragraphs) > 2:
                        third_para_sentences = paragraphs[2].split('。')
                        if len(third_para_sentences) > 2:
                            short_intro += '\n\n' + '。'.join(third_para_sentences[:2]) + '。'
                        else:
                            short_intro += '\n\n' + paragraphs[2]
                else:
                    short_intro = intro_text[:500]
                
                print(f"使用 TTS 朗讀 YCM 公司介紹 (長度: {len(short_intro)} 字符)...")
                
                # 初始化 pygame 混音器
                pygame.mixer.init()
                
                # 使用可用的 TTS 服務朗讀介紹
                tts_success = False
                
                # 檢查 TTS 服務是否可用
                if elevenlabs_tts is not None:
                    print("使用 ElevenLabs TTS 服務朗讀...")
                    audio_path = elevenlabs_tts.synthesize_speech(short_intro)
                    if audio_path:
                        pygame.mixer.music.load(audio_path)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                        tts_success = True
                        print("ElevenLabs TTS 朗讀完成")
                
                if not tts_success and inai_tts is not None:
                    print("使用 Inai TTS 服務朗讀...")
                    audio_path = inai_tts.synthesize_speech(short_intro)
                    if audio_path:
                        pygame.mixer.music.load(audio_path)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                        tts_success = True
                        print("Inai TTS 朗讀完成")
                
                if not tts_success and chatgpt_tts is not None:
                    print("使用 ChatGPT TTS 服務朗讀...")
                    audio_path = chatgpt_tts.synthesize_speech(short_intro)
                    if audio_path:
                        pygame.mixer.music.load(audio_path)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                        tts_success = True
                        print("ChatGPT TTS 朗讀完成")
                
                if not tts_success:
                    print("所有 TTS 服務都無法使用，嘗試播放 MP3 文件")
                    # 如果所有 TTS 服務都無法使用，嘗試播放 MP3 文件
                    if os.path.exists(intro_mp3_path):
                        try:
                            pygame.mixer.music.load(intro_mp3_path)
                            pygame.mixer.music.play()
                            
                            # 等待音頻播放完成
                            while pygame.mixer.music.get_busy():
                                time.sleep(0.1)
                                
                            print("MP3 播放完成")
                        except Exception as e:
                            print(f"播放 MP3 文件時出錯: {str(e)}")
            except Exception as e:
                print(f"讀取或朗讀 TXT 文件時出錯: {str(e)}")
                # 如果 TXT 朗讀失敗，嘗試播放 MP3 文件
                if os.path.exists(intro_mp3_path):
                    try:
                        pygame.mixer.init()
                        pygame.mixer.music.load(intro_mp3_path)
                        pygame.mixer.music.play()
                        
                        # 等待音頻播放完成
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                            
                        print("MP3 播放完成")
                    except Exception as e2:
                        print(f"播放 MP3 文件時出錯: {str(e2)}")
            
            print("YCM 公司介紹播放完成，進入互動模式")
        elif os.path.exists(intro_mp3_path):
            # 如果 TXT 文件不存在但 MP3 文件存在
            try:
                pygame.mixer.init()
                pygame.mixer.music.load(intro_mp3_path)
                pygame.mixer.music.play()
                
                # 等待音頻播放完成
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                    
                print("MP3 播放完成")
            except Exception as e:
                print(f"播放 MP3 文件時出錯: {str(e)}")
    except Exception as e:
        print(f"播放 YCM 公司介紹時發生錯誤: {str(e)}")

def realtime_face_recognition(
    args, 
    chat_window, 
    resource_monitor, 
    conversation_memory, 
    llm_service, 
    face_service, 
    speech_recognizer, 
    elevenlabs_tts,
    inai_tts,
    chatgpt_tts,
    langgraph_conversation,
    feature_queue, 
    result_queue,
    rtsp_manager,
    local_cap,
    meeting_recorder
):
    """Real-time face recognition main function"""
    print("啟動即時人臉識別...")
    
    # Load face features
    known_face_data = face_service.load_face_features()
    if not known_face_data:
        print("無法加載人臉特徵，系統將無法識別用戶")

    # 移除原有的攝像頭初始化代碼
    # 改為使用 RTSPManager 來管理串流
    # Initialize variables for main loop
    frame_count = 0
    current_person = None
    employee_cache = {}
    recent_detections = {}
    active_conversations = set()
    prev_gray = None
    motion_threshold = 5
        # Sleep mode variables
    sleep_mode = False
    last_face_position = None
    no_face_counter = 0
    POSITION_THRESHOLD = 50
    NO_FACE_THRESHOLD = 30
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    no_face_counter = 0
    NO_FACE_THRESHOLD = 30
    POSITION_THRESHOLD = 50
    last_face_position = None
    sleep_mode = False
    
    # 創建顯示窗口
    cv2.namedWindow('Face Recognition', cv2.WINDOW_NORMAL)
    
    # Motion detection variables
    prev_gray = None
    motion_threshold = 5.0
    
    # Initialize feature matching worker thread
    def feature_matching_worker():
        while True:
            try:
                if not feature_queue.empty():
                    face_feature = feature_queue.get()
                    best_match, min_distance = face_service.batch_feature_matching(
                        face_feature, known_face_data
                    )
                    result_queue.put((best_match, min_distance))
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"特徵比對錯誤: {e}")
                time.sleep(0.1)
    
    # Start feature matching thread
    feature_thread = threading.Thread(target=feature_matching_worker, daemon=True)
    feature_thread.start()
    
    # Initialize speech recognition
    speech_text_buffer = ""
    last_speech_time = time.time()
    SPEECH_TIMEOUT = 3.0  # 沒有聲音超過 3 秒就停止監聽
    is_listening = False
    current_sentence = ""  # 用於保存當前正在處理的句子
    
    def process_speech_input():
        nonlocal speech_text_buffer, last_speech_time, is_listening, current_sentence
        
        if not speech_recognizer:  # 如果語音識別器未啟用，直接返回
            return
            
        current_time = time.time()
        # 如果正在監聽且超過靜音時間，則處理並發送文字
        if is_listening and current_sentence and (current_time - last_speech_time) >= SPEECH_TIMEOUT:
            print("檢測到靜音，停止監聽...")
            
            if current_person:
                print(f"自動發送語音輸入: {current_sentence}")
                chat_window.input_field.setText(current_sentence)
                chat_window.send_message()
            else:
                print("未檢測到用戶，無法發送語音輸入")
            
            # 重置所有狀態
            current_sentence = ""
            speech_text_buffer = ""
            is_listening = False
    
    def on_speech_detected(text):
        nonlocal speech_text_buffer, last_speech_time, is_listening, current_sentence
        if text:
            text = text.strip()
            
            # 檢查是否包含喚醒詞（不區分大小寫）
            if "emma" in text.lower():
                print("檢測到喚醒詞 'emma'，開始監聽...")
                is_listening = True
                # 清空之前的內容
                current_sentence = ""
                # 移除喚醒詞，只保留實際命令
                command = text.lower().split("emma", 1)[1].strip()
                
                # 檢查是否包含會議相關關鍵詞
                meeting_action = meeting_recorder.detect_meeting_keywords(command)
                if meeting_action:
                    print(f"檢測到會議相關語音命令: {meeting_action}")
                    
                    # 直接處理會議相關命令
                    if meeting_action == "start_recording":
                        success = meeting_recorder.start_recording()
                        if success:
                            response = "已開始錄製會議。請在會議結束時說「結束會議」或「停止錄製」。"
                        else:
                            response = "無法開始錄製會議，可能已經在錄製中或發生錯誤。"
                        chat_window.show_message(response)
                    
                    elif meeting_action == "stop_recording":
                        success, audio_file = meeting_recorder.stop_recording()
                        if success:
                            chat_window.show_message("會議錄製已停止，正在處理錄音...")
                            
                            # 創建背景線程處理錄音
                            def process_recording():
                                nonlocal audio_file
                                
                                # 估計處理時間
                                audio_duration = meeting_recorder.get_audio_duration(audio_file)
                                estimated_time = max(5, int(audio_duration * 0.2))  # 至少5秒，或錄音時長的20%
                                
                                chat_window.show_message(f"預計需要約 {estimated_time} 秒完成處理...")
                                
                                # 顯示轉錄進度
                                chat_window.show_message("正在轉錄音頻... (1/3)")
                                transcription = meeting_recorder.transcribe_audio(audio_file)
                                
                                if transcription:
                                    # 顯示摘要進度
                                    chat_window.show_message("轉錄完成，正在生成摘要... (2/3)")
                                    
                                    # 獲取會議標題和參與者信息
                                    meeting_title = f"會議 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                                    participants = []
                                    if current_person:
                                        participants.append(current_person)
                                    
                                    # 生成摘要
                                    summary = meeting_recorder.generate_summary(transcription, meeting_title, participants)
                                    if summary:
                                        # 顯示導出進度
                                        chat_window.show_message("摘要已生成，正在導出會議記錄... (3/3)")
                                        
                                        # 導出會議記錄
                                        export_path = meeting_recorder.export_meeting_record(meeting_title, transcription, summary, participants)
                                        
                                        if export_path:
                                            chat_window.show_message(f"✅ 會議記錄處理完成！")
                                            chat_window.show_message(f"會議記錄已導出到: {export_path}")
                                            
                                            # 顯示摘要內容並朗讀
                                            chat_window.show_message("以下是會議摘要：")
                                            chat_window.show_message(summary)
                                            
                                            # 使用 TTS 朗讀摘要
                                            if args.tts_service == "elevenlabs":
                                                elevenlabs_tts.generate_and_play_audio(summary)
                                            elif args.tts_service == "inai":
                                                inai_tts.generate_and_play_audio(summary)
                                            elif args.tts_service == "chatgpt":
                                                chatgpt_tts.generate_and_play_audio(summary)
                                        else:
                                            chat_window.show_message("❌ 導出會議記錄失敗。")
                                    else:
                                        chat_window.show_message("❌ 生成摘要失敗。")
                                else:
                                    chat_window.show_message("❌ 轉錄音頻失敗。")
                            
                            # 啟動背景線程
                            threading.Thread(target=process_recording, daemon=True).start()
                        else:
                            chat_window.show_message("停止錄製失敗，可能沒有進行中的錄製。")
                    
                    # 設置當前句子，這樣會顯示在輸入框中
                    current_sentence = command
                    chat_window.input_field.setText(current_sentence)
                    # 立即處理語音輸入
                    if current_sentence:
                        print(f"自動發送語音輸入: {current_sentence}")
                        chat_window.input_field.setText(current_sentence)
                        chat_window.send_message()
                        # 重置所有狀態
                        current_sentence = ""
                        speech_text_buffer = ""
                        is_listening = False
                        return
                elif is_listening:  # 只有在已經喚醒的狀態下才記錄新的輸入
                    # 更新最後一次收到語音的時間
                    last_speech_time = time.time()
                    
                    # 如果新文本比當前句子長，則更新當前句子
                    if len(text) > len(current_sentence):
                        current_sentence = text
                        chat_window.input_field.setText(current_sentence)
    
    # Set speech recognition callback
    if speech_recognizer:
        speech_recognizer.on_speech_detected = on_speech_detected
        
        # Start speech recognition thread
        speech_thread = threading.Thread(target=speech_recognizer.start_listening, daemon=True)
        speech_thread.start()
    
    # Set chat window message handler
    def on_user_message(message):
        nonlocal current_person, employee_cache
        print(f"收到用戶消息: {message}")  # 添加日誌
        
        # 檢查是否包含會議相關關鍵詞，處理會議錄製功能
        meeting_action = meeting_recorder.detect_meeting_keywords(message)
        if meeting_action:
            print(f"檢測到會議相關關鍵詞，執行動作: {meeting_action}")
            
            if meeting_action == "start_recording":
                success = meeting_recorder.start_recording()
                if success:
                    response = "已開始錄製會議。請在會議結束時說「結束會議」或「停止錄製」。"
                else:
                    response = "無法開始錄製會議，可能已經在錄製中或發生錯誤。"
                chat_window.show_message(response)
                return
                
            elif meeting_action == "stop_recording":
                success, audio_file = meeting_recorder.stop_recording()
                if success:
                    chat_window.show_message("會議錄製已停止，正在處理錄音...")
                    
                    # 創建背景線程處理錄音
                    def process_recording():
                        nonlocal audio_file
                        
                        # 估計處理時間（假設每分鐘錄音需要約10秒處理）
                        audio_duration = meeting_recorder.get_audio_duration(audio_file)
                        estimated_time = max(5, int(audio_duration * 0.2))  # 至少5秒，或錄音時長的20%
                        
                        chat_window.show_message(f"預計需要約 {estimated_time} 秒完成處理...")
                        
                        # 顯示轉錄進度
                        chat_window.show_message("正在轉錄音頻... (1/3)")
                        transcription = meeting_recorder.transcribe_audio(audio_file)
                        
                        if transcription:
                            # 顯示摘要進度
                            chat_window.show_message("轉錄完成，正在生成摘要... (2/3)")
                            
                            # 獲取會議標題和參與者信息
                            meeting_title = f"會議 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                            participants = []
                            if current_person:
                                participants.append(current_person)
                            
                            # 生成摘要
                            summary = meeting_recorder.generate_summary(transcription, meeting_title, participants)
                            if summary:
                                # 顯示導出進度
                                chat_window.show_message("摘要已生成，正在導出會議記錄... (3/3)")
                                
                                # 導出會議記錄
                                export_path = meeting_recorder.export_meeting_record(meeting_title, transcription, summary, participants)
                                if export_path:
                                    chat_window.show_message(f"✅ 會議記錄處理完成！")
                                    chat_window.show_message(f"會議記錄已導出到: {export_path}")
                                    
                                    # 顯示摘要內容並朗讀
                                    chat_window.show_message("以下是會議摘要：")
                                    chat_window.show_message(summary)
                                    
                                    # 使用 TTS 朗讀摘要
                                    if args.tts_service == "elevenlabs":
                                        elevenlabs_tts.generate_and_play_audio(summary)
                                    elif args.tts_service == "inai":
                                        inai_tts.generate_and_play_audio(summary)
                                    elif args.tts_service == "chatgpt":
                                        chatgpt_tts.generate_and_play_audio(summary)
                                else:
                                    chat_window.show_message("❌ 導出會議記錄失敗。")
                            else:
                                chat_window.show_message("❌ 生成摘要失敗。")
                        else:
                            chat_window.show_message("❌ 轉錄音頻失敗。")
                    
                    # 啟動背景線程
                    threading.Thread(target=process_recording, daemon=True).start()
                else:
                    chat_window.show_message("停止錄製失敗，可能沒有進行中的錄製。")
                return
                
            elif meeting_action == "upload_audio":
                chat_window.show_message("請稍等，正在準備音頻上傳功能...")
                # 這裡可以添加打開文件選擇對話框的代碼
                # 或者提示用戶將音頻文件放到特定位置
                chat_window.show_message("請將音頻文件放到 'uploads' 文件夾，然後告訴我文件名。")
                return
                
            elif meeting_action == "process_audio":
                # 假設用戶在消息中提供了文件名
                file_name = message.split("處理音頻")[-1].strip()
                if not file_name:
                    chat_window.show_message("請提供音頻文件名。")
                    return
                
                audio_path = os.path.join("uploads", file_name)
                if not os.path.exists(audio_path):
                    chat_window.show_message(f"找不到音頻文件: {audio_path}")
                    return
                
                chat_window.show_message(f"正在處理音頻文件: {file_name}...")
                
                # 處理音頻文件
                transcription = meeting_recorder.transcribe_audio(audio_path)
                if transcription:
                    chat_window.show_message("轉錄完成，正在生成摘要...")
                    
                    # 獲取會議標題和參與者信息
                    meeting_title = f"會議 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    participants = []
                    if current_person:
                        participants.append(current_person)
                    
                    # 生成摘要
                    summary = meeting_recorder.generate_summary(transcription, meeting_title, participants)
                    if summary:
                        chat_window.show_message("會議摘要已生成，正在導出...")
                        
                        # 導出會議記錄
                        export_path = meeting_recorder.export_meeting_record(meeting_title, transcription, summary, participants)
                        if export_path:
                            chat_window.show_message(f"✅ 會議記錄處理完成！")
                            chat_window.show_message(f"會議記錄已導出到: {export_path}")
                            
                            # 顯示摘要內容並朗讀
                            chat_window.show_message("以下是會議摘要：")
                            chat_window.show_message(summary)
                            
                            # 使用 TTS 朗讀摘要
                            if args.tts_service == "elevenlabs":
                                elevenlabs_tts.generate_and_play_audio(summary)
                            elif args.tts_service == "inai":
                                inai_tts.generate_and_play_audio(summary)
                            elif args.tts_service == "chatgpt":
                                chatgpt_tts.generate_and_play_audio(summary)
                        else:
                            chat_window.show_message("❌ 導出會議記錄失敗。")
                    else:
                        chat_window.show_message("❌ 生成摘要失敗。")
                else:
                    chat_window.show_message("❌ 轉錄音頻失敗。")
                return
        
        # 測試直接回應，檢查界面更新是否正常
        chat_window.show_message("系統啟動測試：請問您能看到這條消息嗎？")
        
        # 正常回應邏輯
        if current_person and current_person in employee_cache:
            try:
                # 檢查是否需要搜索最新資訊
                need_search = check_if_need_search(message)
                search_results = None
                
                # 如果需要搜索，使用 Tavily 服務
                if need_search:
                    try:
                        from services.tavily_service import TavilyService
                        tavily_service = TavilyService()
                        if tavily_service.initialized:
                            print(f"使用 Tavily 搜索: '{message}'")
                            search_result = tavily_service.search(message)
                            print(f"Tavily 原始回應: {search_result}")
                            if search_result and "results" in search_result:
                                search_results = search_result["results"]
                                print(f"搜索結果: {len(search_results)} 項")
                                # 打印搜索結果摘要
                                for i, result in enumerate(search_results[:2], 1):
                                    print(f"  結果 {i}: {result.get('title', '無標題')[:50]}...")
                            else:
                                print("未找到搜索結果")
                    except Exception as e:
                        print(f"Tavily 搜索錯誤: {e}")
                
                # 使用 MongoDB 記憶代理處理消息
                try:
                    response_data = response_generate(employee_cache[current_person], message)
                    response = response_data["content"]
                    print(f"使用模型: {response_data.get('model', '未知')}")
                    if "memory_references" in response_data and response_data["memory_references"]:
                        print(f"參考記憶: {len(response_data['memory_references'])} 條")
                except Exception as e:
                    print(f"MongoDB 記憶代理處理消息時出錯: {str(e)}")
                    # 如果 MongoDB 記憶代理失敗，回退到傳統方式
                    response = llm_service.handle_user_message_with_search(
                        employee_cache[current_person],
                        message,
                        conversation_memory,
                        search_results
                    )
                
                print(f"AI 回應: {response}")
                chat_window.show_message(response)
                
                # 使用 Inai TTS 或 ElevenLabs TTS 合成語音
                if chatgpt_tts and args.use_voice:
                    audio_path = chatgpt_tts.synthesize_speech(response)
                    if audio_path:
                        print(f"語音合成完成，保存到: {audio_path}")
                        # 播放語音
                        import pygame
                        pygame.mixer.init()
                        pygame.mixer.music.load(audio_path)
                        pygame.mixer.music.play()
                elif inai_tts and args.use_voice:
                    audio_path = inai_tts.synthesize_speech(response)
                    if audio_path:
                        print(f"語音合成完成，保存到: {audio_path}")
                        # 播放語音
                        import pygame
                        pygame.mixer.init()
                        pygame.mixer.music.load(audio_path)
                        pygame.mixer.music.play()
                elif elevenlabs_tts and args.use_voice:
                    audio_path = elevenlabs_tts.synthesize_speech(response)
                    if audio_path:
                        print(f"語音合成完成，保存到: {audio_path}")
                        # 播放語音
                        import pygame
                        pygame.mixer.init()
                        pygame.mixer.music.load(audio_path)
                        pygame.mixer.music.play()
            except Exception as e:
                print(f"處理消息時出錯: {e}")
                chat_window.show_message(f"處理消息時出錯: {e}")
        else:
            chat_window.show_message("抱歉，我現在無法確定你是誰。請讓我看清你的臉。")
    
    # 檢查是否需要搜索最新資訊
    def check_if_need_search(message):
        # 檢查是否包含問題關鍵詞
        question_keywords = ["什麼", "如何", "為什麼", "怎麼", "哪裡", "誰", "何時", "多少"]
        
        # 檢查是否包含搜索關鍵詞或時間關鍵詞
        search_keywords = ["最新", "新聞", "資訊", "消息", "查詢", "搜索", "了解", "資料", "現況", "情況", "發展", "技術", "科技", "AI", "人工智能"]
        time_keywords = ["今天", "現在", "最近", "昨天", "明天", "本週", "本月", "今年", "2025", "2024", "未來"]
        
        # 檢查是否是一個問句
        is_question = "?" in message or "？" in message or any(kw in message for kw in question_keywords)
        
        # 檢查是否包含搜索關鍵詞或時間關鍵詞
        has_search_intent = any(kw in message for kw in search_keywords)
        has_time_intent = any(kw in message for kw in time_keywords)
        
        # 如果是問句並且包含搜索關鍵詞或時間關鍵詞，則認為有搜索意圖
        should_search = is_question and (has_search_intent or has_time_intent)
        
        # 如果消息中明確提到 2025 年或未來，強制啟用搜索
        if "2025" in message or "未來" in message or "將來" in message:
            should_search = True
            print("檢測到未來相關關鍵詞，強制啟用搜索")
            
        # 如果消息中提到 AI 或技術發展，也啟用搜索
        if ("AI" in message or "人工智能" in message or "技術" in message or "科技" in message) and ("發展" in message or "趨勢" in message):
            should_search = True
            print("檢測到技術發展相關關鍵詞，強制啟用搜索")
        
        print(f"檢查搜索意圖: '{message}' -> {should_search} (問句: {is_question}, 搜索關鍵詞: {has_search_intent}, 時間關鍵詞: {has_time_intent})")
        
        return should_search
    
    # 確保正確連接信號
    chat_window.message_sent.connect(on_user_message)
    
    print("即時人臉識別系統已啟動...")

    # Main loop
    while True:
        try:
            loop_start = time.time()
            
            if args.no_rtsp and local_cap:
                # 讀取本地攝像頭畫面
                ret, frame = local_cap.read()
                if not ret:
                    print("無法讀取攝像頭畫面")
                    break
                
                # 調整幀大小
                frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
                
                # 更新 FPS 計數器
                fps_counter += 1
                if time.time() - fps_start_time >= 5:
                    current_fps = fps_counter / (time.time() - fps_start_time)
                    print(f"目前 FPS: {current_fps:.1f}, CPU 使用率: {resource_monitor.current_cpu_percent:.1f}%")
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # 增加幀計數
                frame_count += 1
                
                # Skip processing based on CPU usage
                if not resource_monitor.should_process_frame(frame_count):
                    if current_person:
                        # If a user was previously identified, display the name
                        cv2.putText(
                            frame,
                            f"Identified: {current_person}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2
                        )
                    
                    cv2.imshow('Face Recognition', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                # Simple motion detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                motion_detected = False
                if prev_gray is not None:
                    frame_diff = cv2.absdiff(gray, prev_gray)
                    motion_score = np.mean(frame_diff)
                    motion_detected = motion_score > motion_threshold
                
                prev_gray = gray.copy()
                
                # Only detect faces when motion is detected or periodically
                faces = []
                if motion_detected or frame_count % 15 == 0:
                    faces = face_service.detect_faces(frame)
                
                # Update no face counter
                if not faces:
                    no_face_counter += 1
                    if no_face_counter >= NO_FACE_THRESHOLD:
                        sleep_mode = False
                        last_face_position = None
                        current_person = None
                else:
                    no_face_counter = 0
                
                current_time = datetime.now().strftime("%H:%M:%S")
                
                # 處理檢測到的人臉
                if faces:
                    face = faces[0]
                    
                    # 檢查 face 的結構類型，根據不同的結構獲取資訊
                    if hasattr(face, 'bbox') and hasattr(face, 'normed_embedding'):
                        # InsightFace 直接返回的物件
                        current_pos = (face.bbox[0], face.bbox[1])
                        face_feature = face.normed_embedding
                        bbox = face.bbox.astype(int)
                    elif isinstance(face, dict):
                        # 以字典形式封裝的物件
                        current_pos = (face['bbox'][0], face['bbox'][1])
                        face_feature = face['embedding'] if 'embedding' in face else None
                        bbox = face['bbox'].astype(int) if 'bbox' in face else np.array([0, 0, 100, 100])
                    else:
                        # 未能識別的結構，使用默認值
                        current_pos = (0, 0)
                        face_feature = None
                        bbox = np.array([0, 0, 100, 100])
                        print("警告: 無法識別人臉結構類型")
                    
                    last_face_position = current_pos
                    
                    # 提取人臉特徵
                    if face_feature is not None:
                        # 添加到特徵比對隊列
                        if feature_queue.qsize() < 5:
                            feature_queue.put(face_feature)
                    
                    # 繪製人臉框
                    color = (0, 255, 0) if current_person else (0, 165, 255)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    
                    if current_person:
                        # 在人臉上方顯示名字
                        cv2.putText(
                            frame,
                            current_person,
                            (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            color,
                            2
                        )
                
                # 顯示系統資訊
                cv2.putText(
                    frame,
                    f"CPU: {resource_monitor.current_cpu_percent:.1f}%",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    frame,
                    f"FPS: {current_fps:.1f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # 顯示幀
                cv2.imshow('Face Recognition', frame)
            
            elif rtsp_manager:
                # 處理 RTSP 串流的邏輯
                stream_ids = rtsp_manager.get_all_streams()
                if not stream_ids:
                    time.sleep(0.1)
                    continue
                
                # 處理每個串流
                for stream_id in stream_ids:
                    frame = rtsp_manager.get_frame(stream_id)
                    if frame is None:
                        continue
                    
                    # 調整幀大小
                    frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
                    
                    # 處理幀
                    #frame = process_frame(frame)
                    
                    # 顯示串流 ID
                    cv2.putText(
                        frame,
                        f"Stream: {stream_id}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                    
                    # 顯示幀
                    cv2.imshow('Face Recognition', frame)
            
            # 處理每個串流
            # all_faces = []
            # for stream_id in stream_ids:
            #     frame = rtsp_manager.get_frame(stream_id)
            #     if frame is None:
            #         continue
                
            #     # 調整幀大小
            #     frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))

                # Update FPS counter
                fps_counter += 1
                if time.time() - fps_start_time >= 5:
                    current_fps = fps_counter / (time.time() - fps_start_time)
                    print(f"目前 FPS: {current_fps:.1f}, CPU 使用率: {resource_monitor.current_cpu_percent:.1f}%")
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Increment frame count
                frame_count += 1
                
                # Skip processing based on CPU usage
                if not resource_monitor.should_process_frame(frame_count):
                    if current_person:
                        # If a user was previously identified, display the name
                        cv2.putText(
                            frame,
                            f"Identified: {current_person}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2
                        )
                    
                    cv2.imshow('Face Recognition', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                # Simple motion detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                motion_detected = False
                if prev_gray is not None:
                    frame_diff = cv2.absdiff(gray, prev_gray)
                    motion_score = np.mean(frame_diff)
                    motion_detected = motion_score > motion_threshold
                
                prev_gray = gray.copy()
                
                # Only detect faces when motion is detected or periodically
                faces = []
                if motion_detected or frame_count % 15 == 0:
                    faces = face_service.detect_faces(frame)
                
                # Update no face counter
                if not faces:
                    no_face_counter += 1
                    if no_face_counter >= NO_FACE_THRESHOLD:
                        sleep_mode = False
                        last_face_position = None
                        current_person = None
                
                    cv2.imshow('Face Recognition', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                else:
                    no_face_counter = 0
                
                current_time = datetime.now().strftime("%H:%M:%S")
                
                # In sleep mode, only check for position changes
                if sleep_mode and faces:
                    face = faces[0]
                    
                    # 檢查 face 的結構類型，根據不同的結構獲取 bbox
                    if hasattr(face, 'bbox'):
                        # InsightFace 直接返回的物件
                        current_pos = (face.bbox[0], face.bbox[1])
                    elif isinstance(face, dict):
                        # 以字典形式封裝的物件
                        current_pos = (face['bbox'][0], face['bbox'][1])
                    else:
                        # 未能識別的結構，使用默認值
                        current_pos = (0, 0)
                        print("警告: 無法識別人臉結構類型")
                    
                    if last_face_position:
                        dx = current_pos[0] - last_face_position[0]
                        dy = current_pos[1] - last_face_position[1]
                        distance = np.sqrt(dx*dx + dy*dy)
                        
                        if distance > POSITION_THRESHOLD:
                            print("檢測到顯著移動，退出休眠模式")
                            sleep_mode = False
                
                    last_face_position = current_pos
                    
                    # Use the last identified person in sleep mode
                    for person_id, last_time in recent_detections.items():
                        if (datetime.strptime(current_time, "%H:%M:%S") - 
                            datetime.strptime(last_time, "%H:%M:%S")).total_seconds() < 30:
                            current_person = person_id
                            break
            
                # Full face recognition in non-sleep mode
                if not sleep_mode and faces:
                    face = faces[0]
                    
                    # 檢查 face 的結構類型，根據不同的結構獲取資訊
                    if hasattr(face, 'bbox') and hasattr(face, 'normed_embedding'):
                        # InsightFace 直接返回的物件
                        current_pos = (face.bbox[0], face.bbox[1])
                        face_feature = face.normed_embedding
                        bbox = face.bbox.astype(int)
                    elif isinstance(face, dict):
                        # 以字典形式封裝的物件
                        current_pos = (face['bbox'][0], face['bbox'][1])
                        face_feature = face['embedding'] if 'embedding' in face else None
                        bbox = face['bbox'].astype(int) if 'bbox' in face else np.array([0, 0, 100, 100])
                    else:
                        # 未能識別的結構，使用默認值
                        current_pos = (0, 0)
                        face_feature = None
                        bbox = np.array([0, 0, 100, 100])
                        print("警告: 無法識別人臉結構類型")
                    
                    last_face_position = current_pos
                    
                    # Extract face feature
                    if face_feature is not None:
                        # Add to feature matching queue
                        if feature_queue.qsize() < 5:
                            feature_queue.put(face_feature)
            # Check for matching results
            if not result_queue.empty():
                best_match, min_distance = result_queue.get()
                if best_match and min_distance < 0.3:
                    #print(f"識別到用戶: {best_match}, 距離: {min_distance:.4f}")
                    current_person = best_match
                    current_time = datetime.now().strftime("%H:%M:%S")
                    recent_detections[current_person] = current_time
                    
                    # 如果是新用戶，獲取員工資料
                    if current_person not in employee_cache:
                        try:
                            print(f"嘗試獲取員工資料: {current_person}")
                            employee_data = get_employee_data(current_person)
                            if employee_data:
                                print(f"成功獲取員工資料: {employee_data['name']}")
                                employee_cache[current_person] = employee_data
                                
                                # 如果是新對話
                                if current_person not in active_conversations:
                                    print("開始新對話")
                                    
                                    # Start a new conversation in a separate thread
                                    def start_conversation():
                                        # 嘗試使用 LangGraph 處理初始對話
                                        if langgraph_conversation and langgraph_conversation.initialized:
                                            print("使用 LangGraph 開始新對話...")
                                            response, audio_path = langgraph_conversation.process_message(
                                                "你好",  # 初始問候
                                                employee_data,
                                                current_person,
                                                is_first_chat=True
                                            )
                                            if response:
                                                chat_window.show_message(response)
                                            
                                                # 播放語音（如果有）
                                                if audio_path and args.use_voice:
                                                    print(f"播放語音: {audio_path}")
                                                    import pygame
                                                    pygame.mixer.init()
                                                    pygame.mixer.music.load(audio_path)
                                                    pygame.mixer.music.play()
                                        else:
                                            # 使用傳統方式處理初始對話
                                            response = llm_service.chat_with_employee(
                                                employee_data,
                                                is_first_chat=True
                                            )
                                            if response:
                                                chat_window.show_message(response)
                                            
                                                # 使用 Inai TTS 或 ElevenLabs TTS 合成語音
                                                if inai_tts and args.use_voice:
                                                    audio_path = inai_tts.synthesize_speech(response)
                                                    if audio_path:
                                                        print(f"語音合成完成，保存到: {audio_path}")
                                                        # 播放語音
                                                        import pygame
                                                        pygame.mixer.init()
                                                        pygame.mixer.music.load(audio_path)
                                                        pygame.mixer.music.play()
                                                elif elevenlabs_tts and args.use_voice:
                                                    audio_path = elevenlabs_tts.synthesize_speech(response)
                                                    if audio_path:
                                                        print(f"語音合成完成，保存到: {audio_path}")
                                                        # 播放語音
                                                        import pygame
                                                        pygame.mixer.init()
                                                        pygame.mixer.music.load(audio_path)
                                                        pygame.mixer.music.play()
                                                elif chatgpt_tts and args.use_voice:
                                                    audio_path = chatgpt_tts.synthesize_speech(response)
                                                    if audio_path:
                                                        print(f"語音合成完成，保存到: {audio_path}")
                                                        # 播放語音
                                                        import pygame
                                                        pygame.mixer.init()
                                                        pygame.mixer.music.load(audio_path)
                                                        pygame.mixer.music.play()
                                    
                                    threading.Thread(target=start_conversation, daemon=True).start()
                                    active_conversations.add(current_person)
                                    print("開始新對話")
                        except Exception as e:
                            print(f"獲取員工資料時發生錯誤: {e}")
                            
                    # Generate conversation summary every 10 minutes
                    if current_person in active_conversations:
                        current_minute = datetime.now().minute
                        if current_minute % 10 == 0 and current_minute != 0:
                            threading.Thread(
                                target=conversation_memory.generate_conversation_summary,
                                args=(current_person, llm_service),
                                daemon=True
                            ).start()
                else:
                    #print(f"無法識別用戶，最小距離: {min_distance:.4f}")
                    # Only reset current user when distance is very large
                    if min_distance > 0.6:
                        current_person = None
            
            # Process speech input
            process_speech_input()
            
            # Draw faces on frame
            if faces:
                face = faces[0]
                
                # 檢查 face 的結構類型，根據不同的結構獲取 bbox
                if hasattr(face, 'bbox'):
                    # InsightFace 直接返回的物件
                    bbox = face.bbox.astype(int)
                elif isinstance(face, dict) and 'bbox' in face:
                    # 以字典形式封裝的物件
                    bbox = face['bbox'].astype(int)
                else:
                    # 未能識別的結構，使用默認值
                    bbox = np.array([0, 0, 100, 100])
                    print("警告: 無法識別人臉結構類型")
                
                # Choose color based on recognition result
                if current_person:
                    color = (0, 255, 0)  # Green - identified
                else:
                    color = (0, 165, 255)  # Orange - unidentified
                
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                if current_person:
                    # Display name above the face
                    cv2.putText(
                        frame,
                        current_person,
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2
                    )
                    
                    # Display system info in corner
                    cv2.putText(
                        frame,
                        f"CPU: {resource_monitor.current_cpu_percent:.1f}%",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"FPS: {current_fps:.1f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
            
            # Calculate and display processing time
            process_time = time.time() - loop_start
            if process_time > 0.1:
                print(f"警告: 處理時間較長 {process_time:.3f}秒")
            
            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # 控制處理速率
            time.sleep(0.03)
            
        except Exception as e:
            print(f"處理影像時發生錯誤: {str(e)}")
            time.sleep(0.1)
            continue
    
    # 清理資源
    if args.no_rtsp:
        if local_cap is not None:
            local_cap.release()
    else:
        if rtsp_manager is not None:
            for stream_id in rtsp_manager.get_all_streams():
                rtsp_manager.remove_stream(stream_id)
    
    cv2.destroyAllWindows()
    resource_monitor.stop_monitoring()
    if speech_recognizer:
        speech_recognizer.stop_listening()
    
    print("即時人臉識別系統已關閉")

def train_face(name=None):
    """訓練新的人臉特徵"""
    if name is None:
        name = input("請輸入人名: ")
    
    print(f"開始為 {name} 訓練人臉特徵...")
    
    # 初始化攝像頭
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("錯誤：無法開啟攝像頭")
        return
    
    # 初始化人臉識別服務
    face_service = FaceService()
    
    # 載入現有的人臉特徵
    face_features = face_service.load_face_features()
    if name not in face_features:
        face_features[name] = []
    
    try:
        # 創建窗口
        cv2.namedWindow('Training', cv2.WINDOW_NORMAL)
        print("請看著攝像頭，系統會自動收集不同角度的人臉特徵")
        print("按空格鍵保存當前特徵，按 ESC 結束訓練")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("無法讀取攝像頭畫面")
                break
            
            # 檢測人臉
            faces = face_service.detect_faces(frame)
            
            if faces:
                # 顯示檢測到的人臉
                for face in faces:
                    if hasattr(face, 'bbox'):
                        bbox = face.bbox.astype(int)
                    elif isinstance(face, dict) and 'bbox' in face:
                        bbox = face['bbox'].astype(int)
                    else:
                        continue
                    
                    # 繪製人臉框
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # 顯示畫面
            cv2.imshow('Training', frame)
            
            # 檢查按鍵
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # 空格
                if faces:
                    face = faces[0]
                    if hasattr(face, 'normed_embedding'):
                        face_feature = face.normed_embedding
                    elif isinstance(face, dict) and 'embedding' in face:
                        face_feature = face['embedding']
                    else:
                        continue
                    
                    if face_feature is not None:
                        face_features[name].append(face_feature.tolist())
                        print(f"已保存第 {len(face_features[name])} 個特徵")
                else:
                    print("未檢測到人臉，請調整位置")
    
    finally:
        # 保存特徵
        if len(face_features[name]) > 0:
            face_service.save_face_features(face_features)
            print(f"已保存 {len(face_features[name])} 個特徵到 face_features.json")
        else:
            print("未收集到任何特徵")
        
        # 釋放資源
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        # 訓練模式
        if len(sys.argv) > 2:
            train_face(sys.argv[2])
        else:
            train_face()
    else:
        # 正常運行模式
        main()