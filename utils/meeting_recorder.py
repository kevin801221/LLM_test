"""
Meeting recorder integration module.
This module integrates the YCM meeting record functionality into the Emma chatbot system.
"""

import os
import sys
import time
import json
import requests
import datetime
import re
import wave
import pyaudio
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import openai
from dotenv import load_dotenv

# 加載環境變量
load_dotenv()

# 獲取 OpenAI API 密鑰
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

class AudioRecorder:
    """Audio recorder class for recording meeting audio."""
    
    def __init__(self):
        """Initialize the audio recorder."""
        self.is_recording = False
        self.frames = []
        self.stream = None
        self.audio = None
        self.temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp_audio")
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        
        # Audio settings
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.chunk = 1024
    
    def start_recording(self):
        """Start recording audio."""
        if self.is_recording:
            return "已經在錄音中"
        
        try:
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            self.frames = []
            self.is_recording = True
            
            # Start recording in a separate thread
            import threading
            self.recording_thread = threading.Thread(target=self._record)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            return "錄音開始"
        except Exception as e:
            return f"錄音開始失敗: {str(e)}"
    
    def _record(self):
        """Record audio data."""
        while self.is_recording:
            try:
                data = self.stream.read(self.chunk)
                self.frames.append(data)
            except Exception as e:
                print(f"錄音錯誤: {str(e)}")
                break
    
    def stop_recording(self):
        """Stop recording audio."""
        if not self.is_recording:
            return None, "沒有進行中的錄音"
        
        self.is_recording = False
        
        # Wait for recording thread to finish
        if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=1.0)
        
        # Stop and close the stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        # Terminate the PyAudio object
        if self.audio:
            self.audio.terminate()
        
        # Save the recorded audio to a WAV file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.temp_dir, f"meeting_recording_{timestamp}.wav")
        
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(self.frames))
            
            return filename, f"錄音已保存到 {filename}"
        except Exception as e:
            return None, f"保存錄音失敗: {str(e)}"
    
    def get_audio_duration(self, audio_file):
        """Get the duration of an audio file in seconds."""
        if not os.path.exists(audio_file):
            return 0
        
        try:
            with wave.open(audio_file, 'rb') as wf:
                # 計算音頻時長（秒）
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
                return duration
        except Exception as e:
            print(f"獲取音頻時長時出錯: {str(e)}")
            return 0


class Transcriber:
    """Transcriber class for transcribing audio to text."""
    
    def __init__(self):
        """Initialize the transcriber."""
        self.transcriptions = []
    
    def transcribe_audio(self, audio_file: str) -> str:
        """Transcribe audio file to text using OpenAI's Whisper model."""
        if not os.path.exists(audio_file):
            return f"找不到音頻文件: {audio_file}"
        
        try:
            with open(audio_file, "rb") as file:
                transcription = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=file,
                    language="zh"
                )
            
            text = transcription.text
            self.transcriptions.append(text)
            return text
        except Exception as e:
            return f"轉錄失敗: {str(e)}"
    
    def get_all_transcriptions(self) -> List[str]:
        """Get all transcriptions."""
        return self.transcriptions


class SummaryGenerator:
    """Summary generator class for generating meeting summaries."""
    
    def __init__(self):
        """Initialize the summary generator."""
        self.ollama_api_url = "http://localhost:11434/api/generate"
    
    def clean_summary(self, summary: str) -> str:
        """Clean the summary by removing special tokens and markdown formatting."""
        # 移除特殊標記，如 <think>、</think> 等
        summary = re.sub(r'<[^>]+>', '', summary)
        
        # 移除 Markdown 格式符號
        summary = re.sub(r'\*\*|\*|##|###', '', summary)
        
        # 移除多餘的空行
        summary = re.sub(r'\n{3,}', '\n\n', summary)
        
        return summary.strip()
    
    def generate_summary(self, transcription: str, meeting_title: str = "", participants: list = None) -> str:
        """Generate a summary from the meeting transcription using local Gemma model."""
        if not transcription:
            return "沒有可用的會議轉錄內容"
        
        if participants is None:
            participants = []
        
        participants_str = ", ".join(participants) if participants else "未知"
        
        # 構建提示
        prompt = f"""
你是一個專業的會議摘要生成助手。請根據以下會議轉錄內容，生成一個簡潔、結構化的會議摘要。
不要使用任何特殊標記（如 <think>）或 Markdown 格式（如 **、##）。

會議標題: {meeting_title}
參與者: {participants_str}

會議轉錄內容:
{transcription}

請生成以下內容:
1. 會議摘要：概述會議的主要目的和內容
2. 主要討論點：列出會議中討論的主要話題
3. 決策：列出會議中達成的決策
4. 行動項目：列出需要採取的後續行動，包括負責人和時間表（如有提及）

請使用清晰、簡潔的語言，不要添加任何未在轉錄中提及的內容。
"""
        
        try:
            # 嘗試使用本地 Ollama API 生成摘要
            payload = {
                "model": "gemma:12b",
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(self.ollama_api_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                summary = result.get("response", "")
                
                # 清理摘要
                cleaned_summary = self.clean_summary(summary)
                return cleaned_summary
            else:
                # 如果本地模型失敗，嘗試使用 OpenAI API
                return self._generate_summary_with_openai(prompt)
        except Exception as e:
            # 如果本地模型失敗，嘗試使用 OpenAI API
            print(f"使用本地模型生成摘要失敗: {str(e)}，嘗試使用 OpenAI API")
            return self._generate_summary_with_openai(prompt)
    
    def _generate_summary_with_openai(self, prompt: str) -> str:
        """Generate summary using OpenAI API as fallback."""
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一個專業的會議摘要生成助手。"},
                    {"role": "user", "content": prompt}
                ]
            )
            
            summary = response.choices[0].message.content
            return self.clean_summary(summary)
        except Exception as e:
            return f"生成摘要失敗: {str(e)}"


class Exporter:
    """Exporter class for exporting meeting records."""
    
    def __init__(self):
        """Initialize the exporter."""
        self.export_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "meeting_exports")
        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)
    
    def export_meeting(self, title: str, participants: List[str], transcriptions: List[str], summary: str) -> str:
        """Export meeting record to a file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = re.sub(r'[\\/*?:"<>|]', "_", title)  # 替換不安全的文件名字符
        
        filename = os.path.join(self.export_dir, f"{safe_title}_{timestamp}.txt")
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"會議標題: {title}\n")
                f.write(f"日期時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"參與者: {', '.join(participants)}\n")
                f.write("\n" + "="*50 + "\n\n")
                
                f.write("會議摘要:\n")
                f.write(summary)
                f.write("\n\n" + "="*50 + "\n\n")
                
                f.write("會議轉錄:\n")
                for i, transcription in enumerate(transcriptions, 1):
                    f.write(f"--- 部分 {i} ---\n")
                    f.write(transcription)
                    f.write("\n\n")
            
            return f"會議記錄已導出到: {filename}"
        except Exception as e:
            return f"導出會議記錄失敗: {str(e)}"


class MeetingRecorderIntegration:
    """Integration class for meeting recorder functionality."""
    
    def __init__(self):
        """Initialize the meeting recorder integration."""
        self.audio_recorder = AudioRecorder()
        self.transcriber = Transcriber()
        self.summary_generator = SummaryGenerator()
        self.exporter = Exporter()
        
        self.is_recording = False
        self.recording_start_time = None
        self.meeting_title = f"會議 {datetime.datetime.now().strftime('%Y-%m-%d')}"
        self.participants = []
        self.current_audio_file = None
        self.last_transcription = ""
        self.last_summary = ""
        
        # Create a directory for temporary audio files if it doesn't exist
        self.temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp_audio")
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
    
    def start_recording(self) -> bool:
        """Start recording the meeting."""
        if self.is_recording:
            return False
        
        self.is_recording = True
        self.recording_start_time = datetime.datetime.now()
        self.audio_recorder.start_recording()
        return True
    
    def stop_recording(self) -> Tuple[bool, str]:
        """Stop recording the meeting."""
        if not self.is_recording:
            return False, None
        
        self.is_recording = False
        audio_file, _ = self.audio_recorder.stop_recording()
        self.current_audio_file = audio_file
        return True, audio_file
    
    def transcribe_audio(self, audio_file: str) -> str:
        """Transcribe the audio file."""
        transcription = self.transcriber.transcribe_audio(audio_file)
        self.last_transcription = transcription
        return transcription
    
    def generate_summary(self, transcription: str, meeting_title: str = "", participants: list = None) -> str:
        """Generate a summary for the transcription."""
        if not meeting_title:
            meeting_title = self.meeting_title
        
        if participants is None:
            participants = self.participants
        
        summary = self.summary_generator.generate_summary(transcription, meeting_title, participants)
        self.last_summary = summary
        return summary
    
    def export_meeting_record(self, title: str, transcription: str, summary: str, participants: list = None) -> str:
        """Export the meeting record."""
        if participants is None:
            participants = self.participants
        
        export_status = self.exporter.export_meeting(
            title,
            participants,
            [transcription],
            summary
        )
        return export_status
    
    def detect_meeting_keywords(self, message: str) -> Optional[str]:
        """Detect meeting-related keywords in a message."""
        message = message.lower()
        
        # Define keywords for different actions
        start_keywords = ["開始會議", "開始錄音", "錄製會議", "記錄會議", "會議開始", "開始"]
        stop_keywords = ["結束會議", "停止錄音", "會議結束", "停止會議", "結束", "停止"]
        upload_keywords = ["上傳音頻", "上傳錄音", "處理音頻文件"]
        process_keywords = ["處理音頻", "處理錄音"]
        
        # Check for keyword matches
        if any(keyword in message for keyword in start_keywords):
            return "start_recording"
        elif any(keyword in message for keyword in stop_keywords):
            return "stop_recording"
        elif any(keyword in message for keyword in upload_keywords):
            return "upload_audio"
        elif any(keyword in message for keyword in process_keywords):
            return "process_audio"
        
        return None

    def set_meeting_info(self, title: str, participants_str: str) -> str:
        """Set meeting information."""
        self.meeting_title = title if title else self.meeting_title
        self.participants = [p.strip() for p in participants_str.split(",") if p.strip()]
        return f"會議訊息已設置: {self.meeting_title} (參與者: {', '.join(self.participants)})"

    def handle_meeting_command(self, message: str, user_name: str = "", additional_participants: str = "") -> Tuple[bool, str]:
        """Handle meeting-related commands based on message content."""
        # Detect keywords
        keyword_matches = self.detect_meeting_keywords(message)
        
        # If no meeting-related keywords detected
        if not keyword_matches:
            return False, ""
        
        # Handle meeting commands
        if keyword_matches == "start_recording":
            # Extract meeting title from message if available
            title_match = message.split("開始會議")[-1].strip() if "開始會議" in message else ""
            title = title_match if title_match else f"{datetime.datetime.now().strftime('%Y-%m-%d')} 會議"
            
            # Set participants
            participants = user_name
            if additional_participants:
                participants += f", {additional_participants}"
            
            self.set_meeting_info(title, participants)
            response = self.start_recording()
            return True, response
        
        elif keyword_matches == "stop_recording":
            audio_file, response = self.stop_recording()
            if audio_file:
                # Automatically process the meeting after stopping
                transcription = self.transcribe_audio(audio_file)
                summary = self.generate_summary(transcription)
                response += f"\n\n會議已自動處理。\n\n摘要：\n{summary}"
            return True, response
        
        elif keyword_matches == "upload_audio":
            # Handle upload audio command
            pass
        
        elif keyword_matches == "process_audio":
            # Handle process audio command
            pass
        
        return False, ""

    def get_audio_duration(self, audio_file):
        """Get the duration of an audio file in seconds."""
        return self.audio_recorder.get_audio_duration(audio_file)
