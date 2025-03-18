"""
測試會議記錄功能的腳本
"""

import time
import os
from utils.meeting_recorder import MeetingRecorderIntegration

def main():
    # 創建日誌文件
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meeting_test_log.txt")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("會議記錄功能測試日誌\n")
        f.write("=" * 50 + "\n")
    
    def log(message):
        print(message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + "\n")
    
    log("開始測試會議記錄功能...")
    
    # 初始化會議記錄器
    recorder = MeetingRecorderIntegration()
    
    # 設置會議信息
    recorder.set_meeting_info("測試會議", "測試用戶1, 測試用戶2")
    log("會議信息已設置")
    
    # 開始錄音
    log("開始錄音 (將錄製 5 秒)...")
    success = recorder.start_recording()
    if not success:
        log("開始錄音失敗")
        return
    
    # 錄製 5 秒
    time.sleep(5)
    
    # 停止錄音
    log("停止錄音...")
    success, audio_file = recorder.stop_recording()
    if not success or not audio_file:
        log("停止錄音失敗")
        return
    
    log(f"錄音已保存到: {audio_file}")
    
    # 轉錄音頻
    log("開始轉錄音頻...")
    transcription = recorder.transcribe_audio(audio_file)
    log("轉錄結果:")
    log("-" * 50)
    log(transcription)
    log("-" * 50)
    
    # 生成摘要
    log("開始生成摘要...")
    summary = recorder.generate_summary(transcription)
    log("摘要結果:")
    log("-" * 50)
    log(summary)
    log("-" * 50)
    
    # 導出會議記錄
    log("開始導出會議記錄...")
    export_path = recorder.export_meeting_record("測試會議", transcription, summary)
    log(f"導出結果: {export_path}")
    
    # 檢查導出文件的內容
    if os.path.exists(export_path):
        log("導出文件內容:")
        log("-" * 50)
        with open(export_path, 'r', encoding='utf-8') as f:
            log(f.read())
        log("-" * 50)
    
    log("測試完成")
    log(f"詳細日誌已保存到: {log_file}")

if __name__ == "__main__":
    main()
