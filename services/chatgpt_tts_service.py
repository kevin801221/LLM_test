#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import requests
import json
from datetime import datetime
import pygame
import tempfile

class ChatGPTTTSService:
    """Service for handling ChatGPT TTS API interactions"""
    
    def __init__(self, api_url=None, api_key=None, voice="shimmer"):
        """Initialize ChatGPT TTS service
        
        Args:
            api_url: API URL for ChatGPT TTS service
            api_key: API key for ChatGPT TTS service
            voice: Voice to use for TTS
        """
        # 更新為新的 API URL
        self.api_url = api_url or "https://api.openai.com/v1/audio/speech"
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.voice = voice
        self.audio_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio")
        
        # 確保音頻目錄存在
        if not os.path.exists(self.audio_dir):
            os.makedirs(self.audio_dir)
            
        # 初始化 pygame mixer
        pygame.mixer.init()
        
        # 測試連接
        try:
            self._test_connection()
            self.initialized = True
            print(f"ChatGPT TTS 服務初始化成功，使用語音: {self.voice}")
        except Exception as e:
            print(f"ChatGPT TTS 服務初始化失敗: {e}")
            self.initialized = False
    
    def _test_connection(self):
        """Test connection to ChatGPT TTS API"""
        test_text = "測試連接"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "tts-1",
            "input": test_text,
            "voice": self.voice
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                print("ChatGPT TTS API 連接測試成功")
            else:
                print(f"ChatGPT TTS API 連接測試失敗: {response.status_code}, {response.text}")
                raise Exception(f"API 回應錯誤: {response.status_code}")
        except Exception as e:
            print(f"ChatGPT TTS API 連接測試失敗: {e}")
            raise
    
    def synthesize_speech(self, text):
        """Synthesize speech from text
        
        Args:
            text: Text to synthesize
            
        Returns:
            Path to audio file or None if error
        """
        if not self.initialized:
            print("ChatGPT TTS 服務未初始化，無法合成語音")
            return None
            
        print(f"開始合成語音: '{text[:50]}...'")
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "tts-1",
            "input": text,
            "voice": self.voice
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                # 生成文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_file = os.path.join(self.audio_dir, f"tts_{timestamp}.mp3")
                
                # 保存音頻文件
                with open(audio_file, 'wb') as f:
                    f.write(response.content)
                
                print(f"語音合成完成，耗時: {time.time() - start_time:.2f}秒")
                return audio_file
            else:
                print(f"語音合成失敗: {response.status_code}, {response.text}")
                return None
        except Exception as e:
            print(f"語音合成出錯: {e}")
            return None
    
    def play_audio(self, audio_path):
        """Play audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            True if success, False if error
        """
        if not os.path.exists(audio_path):
            print(f"音頻文件不存在: {audio_path}")
            return False
            
        try:
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            return True
        except Exception as e:
            print(f"播放音頻時出錯: {e}")
            return False
