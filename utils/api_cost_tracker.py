#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from datetime import datetime
from functools import wraps
import tiktoken

class APICostTracker:
    """追蹤 OpenAI API 使用成本的工具"""
    
    # OpenAI API 價格（截至 2025 年 3 月）
    PRICING = {
        # GPT 模型
        "gpt-4o": {"input": 0.005, "output": 0.015},  # 每 1K tokens 的美元價格
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
        # 語音模型
        "whisper-1": {"audio_minute": 0.006},  # 每分鐘的美元價格
        # TTS 模型
        "tts-1": {"input": 0.015},  # 每 1K 字符的美元價格
        "tts-1-hd": {"input": 0.03}  # 每 1K 字符的美元價格
    }
    
    def __init__(self, log_file=None):
        """初始化 API 成本追蹤器
        
        Args:
            log_file: 成本日誌文件路徑，如果為 None，則使用默認路徑
        """
        self.log_file = log_file or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "logs", 
            "api_cost.json"
        )
        
        # 確保日誌目錄存在
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # 初始化成本記錄
        self.cost_log = self._load_cost_log()
        
        # 初始化 token 計數器
        self.tokenizers = {}
        
    def _load_cost_log(self):
        """加載成本日誌"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加載成本日誌時出錯: {e}")
                return {"sessions": [], "total_cost": 0.0}
        else:
            return {"sessions": [], "total_cost": 0.0}
    
    def _save_cost_log(self):
        """保存成本日誌"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.cost_log, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存成本日誌時出錯: {e}")
    
    def _get_tokenizer(self, model):
        """獲取指定模型的 tokenizer"""
        if model not in self.tokenizers:
            try:
                # 針對不同模型選擇合適的編碼
                encoding_name = "cl100k_base"  # 默認使用 GPT-4 和 GPT-3.5 Turbo 的編碼
                self.tokenizers[model] = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                print(f"獲取 tokenizer 時出錯: {e}")
                return None
        return self.tokenizers[model]
    
    def _count_tokens(self, text, model):
        """計算文本的 token 數量"""
        tokenizer = self._get_tokenizer(model)
        if tokenizer:
            return len(tokenizer.encode(text))
        # 如果無法獲取 tokenizer，使用簡單估算（每 4 個字符約 1 個 token）
        return len(text) // 4
    
    def track_chat_completion(self, model, messages, response):
        """追蹤聊天完成請求的成本
        
        Args:
            model: 使用的模型名稱
            messages: 發送的消息列表
            response: API 響應對象
        
        Returns:
            dict: 成本信息
        """
        # 標準化模型名稱
        if model == "gpt4o":
            model = "gpt-4o"
        
        # 獲取價格信息
        model_pricing = self.PRICING.get(model, self.PRICING.get("gpt-3.5-turbo"))
        
        # 計算輸入 tokens
        input_tokens = 0
        for msg in messages:
            content = msg.get("content", "")
            input_tokens += self._count_tokens(content, model)
        
        # 獲取輸出 tokens
        try:
            output_tokens = len(response.choices[0].message.content) // 4  # 簡單估算
            if hasattr(response, 'usage') and response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
        except:
            output_tokens = 0
        
        # 計算成本
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]
        total_cost = input_cost + output_cost
        
        # 記錄成本
        cost_info = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "type": "chat_completion"
        }
        
        self._log_cost(cost_info)
        
        return cost_info
    
    def track_audio_transcription(self, model, audio_duration_seconds):
        """追蹤音頻轉錄請求的成本
        
        Args:
            model: 使用的模型名稱
            audio_duration_seconds: 音頻時長（秒）
        
        Returns:
            dict: 成本信息
        """
        # 獲取價格信息
        model_pricing = self.PRICING.get(model, self.PRICING.get("whisper-1"))
        
        # 計算成本
        audio_minutes = audio_duration_seconds / 60
        total_cost = audio_minutes * model_pricing["audio_minute"]
        
        # 記錄成本
        cost_info = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "audio_duration_seconds": audio_duration_seconds,
            "audio_minutes": audio_minutes,
            "total_cost": total_cost,
            "type": "audio_transcription"
        }
        
        self._log_cost(cost_info)
        
        return cost_info
    
    def track_text_to_speech(self, model, text):
        """追蹤文本轉語音請求的成本
        
        Args:
            model: 使用的模型名稱
            text: 輸入文本
        
        Returns:
            dict: 成本信息
        """
        # 獲取價格信息
        model_pricing = self.PRICING.get(model, self.PRICING.get("tts-1"))
        
        # 計算字符數
        char_count = len(text)
        
        # 計算成本
        total_cost = (char_count / 1000) * model_pricing["input"]
        
        # 記錄成本
        cost_info = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "char_count": char_count,
            "total_cost": total_cost,
            "type": "text_to_speech"
        }
        
        self._log_cost(cost_info)
        
        return cost_info
    
    def _log_cost(self, cost_info):
        """記錄成本信息"""
        # 查找當前會話或創建新會話
        today = datetime.now().strftime("%Y-%m-%d")
        current_session = None
        
        for session in self.cost_log["sessions"]:
            if session["date"] == today:
                current_session = session
                break
        
        if not current_session:
            current_session = {
                "date": today,
                "requests": [],
                "session_cost": 0.0
            }
            self.cost_log["sessions"].append(current_session)
        
        # 添加請求記錄
        current_session["requests"].append(cost_info)
        
        # 更新成本
        current_session["session_cost"] += cost_info["total_cost"]
        self.cost_log["total_cost"] += cost_info["total_cost"]
        
        # 保存日誌
        self._save_cost_log()
    
    def get_today_cost(self):
        """獲取今天的成本統計"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        for session in self.cost_log["sessions"]:
            if session["date"] == today:
                return {
                    "date": today,
                    "request_count": len(session["requests"]),
                    "total_cost": session["session_cost"]
                }
        
        return {
            "date": today,
            "request_count": 0,
            "total_cost": 0.0
        }
    
    def get_total_cost(self):
        """獲取總成本統計"""
        return {
            "total_sessions": len(self.cost_log["sessions"]),
            "total_cost": self.cost_log["total_cost"]
        }
    
    def print_cost_summary(self):
        """打印成本摘要"""
        today_cost = self.get_today_cost()
        total_cost = self.get_total_cost()
        
        print("\n===== API 成本摘要 =====")
        print(f"今日 ({today_cost['date']}):")
        print(f"  請求數: {today_cost['request_count']}")
        print(f"  成本: ${today_cost['total_cost']:.6f} USD")
        print("\n總計:")
        print(f"  會話數: {total_cost['total_sessions']}")
        print(f"  總成本: ${total_cost['total_cost']:.6f} USD")
        print("=======================\n")

# 裝飾器函數，用於追蹤 OpenAI API 調用成本
def track_openai_cost(api_type, model=None):
    """裝飾器: 追蹤 OpenAI API 調用成本
    
    Args:
        api_type: API 類型 ('chat', 'audio', 'tts')
        model: 使用的模型名稱，如果為 None，則從函數參數中獲取
    
    Returns:
        裝飾器函數
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 創建成本追蹤器
            cost_tracker = APICostTracker()
            
            # 獲取開始時間
            start_time = datetime.now()
            
            # 調用原始函數
            result = func(*args, **kwargs)
            
            # 計算時間差
            duration = (datetime.now() - start_time).total_seconds()
            
            # 根據 API 類型追蹤成本
            if api_type == 'chat':
                # 從參數中獲取模型名稱
                model_name = model or kwargs.get('model', args[0] if len(args) > 0 else "gpt-3.5-turbo")
                messages = kwargs.get('messages', args[1] if len(args) > 1 else [])
                cost_tracker.track_chat_completion(model_name, messages, result)
            elif api_type == 'audio':
                model_name = model or kwargs.get('model', "whisper-1")
                # 假設音頻文件是第一個參數
                audio_file = kwargs.get('file') or args[0]
                # 估算音頻時長（秒）
                audio_duration = duration  # 使用處理時間作為粗略估計
                cost_tracker.track_audio_transcription(model_name, audio_duration)
            elif api_type == 'tts':
                model_name = model or kwargs.get('model', "tts-1")
                # 假設文本是第一個參數
                text = kwargs.get('text') or args[0]
                cost_tracker.track_text_to_speech(model_name, text)
            
            # 打印成本摘要
            cost_tracker.print_cost_summary()
            
            return result
        return wrapper
    return decorator
