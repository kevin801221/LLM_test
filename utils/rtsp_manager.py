#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import threading
import time
from queue import Queue
from services.common_service import rtsp_event_queue

class RTSPManager:
    def __init__(self):
        self.streams = {}
        self.stream_threads = {}
        self._lock = threading.Lock()
        self.frame_queues = {}  # 用於存儲每個串流的最新幀
        self.frame_queue_size = 2  # 每個串流保持最新的2幀

    def add_stream(self, stream_id, rtsp_url):
        """添加新的 RTSP 串流"""
        with self._lock:
            if stream_id in self.streams:
                print(f"串流 {stream_id} 已存在")
                return False
            
            try:
                cap = cv2.VideoCapture(rtsp_url)
                if not cap.isOpened():
                    print(f"無法連接到串流 {rtsp_url}")
                    return False
                
                # 設置 OpenCV 參數
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 減少緩衝
                
                self.frame_queues[stream_id] = Queue(maxsize=self.frame_queue_size)
                self.streams[stream_id] = {
                    'url': rtsp_url,
                    'cap': cap,
                    'active': True,
                    'last_frame': None,
                    'last_frame_time': time.time()
                }
                
                # 啟動串流處理線程
                thread = threading.Thread(
                    target=self._process_stream,
                    args=(stream_id,),
                    daemon=True
                )
                self.stream_threads[stream_id] = thread
                thread.start()
                
                print(f"成功添加串流 {stream_id}")
                return True
            except Exception as e:
                print(f"添加串流時發生錯誤: {e}")
                if stream_id in self.frame_queues:
                    del self.frame_queues[stream_id]
                return False

    def remove_stream(self, stream_id):
        """移除 RTSP 串流"""
        with self._lock:
            if stream_id not in self.streams:
                return False
            
            print(f"正在移除串流 {stream_id}")
            self.streams[stream_id]['active'] = False
            
            # 清理串流資源
            cap = self.streams[stream_id]['cap']
            cap.release()
            
            # 清理幀隊列
            if stream_id in self.frame_queues:
                while not self.frame_queues[stream_id].empty():
                    try:
                        self.frame_queues[stream_id].get_nowait()
                    except:
                        pass
                del self.frame_queues[stream_id]
            
            # 等待線程結束
            if stream_id in self.stream_threads:
                try:
                    self.stream_threads[stream_id].join(timeout=1.0)
                except:
                    pass
                del self.stream_threads[stream_id]
            
            del self.streams[stream_id]
            print(f"已移除串流 {stream_id}")
            return True

    def _process_stream(self, stream_id):
        """處理單個串流的線程函數"""
        reconnect_delay = 1.0  # 重連延遲時間（秒）
        max_empty_frames = 5   # 最大連續空幀數
        empty_frame_count = 0  # 當前連續空幀計數
        
        while self.streams[stream_id]['active']:
            try:
                cap = self.streams[stream_id]['cap']
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    empty_frame_count += 1
                    print(f"串流 {stream_id} 讀取失敗 ({empty_frame_count}/{max_empty_frames})")
                    
                    if empty_frame_count >= max_empty_frames:
                        print(f"串流 {stream_id} 連續讀取失敗，嘗試重新連接...")
                        cap.release()
                        time.sleep(reconnect_delay)
                        cap = cv2.VideoCapture(self.streams[stream_id]['url'])
                        self.streams[stream_id]['cap'] = cap
                        empty_frame_count = 0
                        continue
                        
                    time.sleep(0.1)
                    continue
                
                empty_frame_count = 0  # 重置空幀計數
                
                # 更新最新幀
                try:
                    if self.frame_queues[stream_id].full():
                        self.frame_queues[stream_id].get_nowait()  # 移除舊幀
                    self.frame_queues[stream_id].put_nowait(frame)
                    self.streams[stream_id]['last_frame'] = frame
                    self.streams[stream_id]['last_frame_time'] = time.time()
                except:
                    pass
                
                time.sleep(0.03)  # 控制處理速率
                
            except Exception as e:
                print(f"處理串流 {stream_id} 時發生錯誤: {str(e)}")
                time.sleep(0.1)

    def get_frame(self, stream_id):
        """獲取指定串流的最新幀"""
        if stream_id not in self.streams:
            return None
            
        try:
            frame = self.frame_queues[stream_id].get_nowait()
            return frame
        except:
            return self.streams[stream_id].get('last_frame')

    def get_all_streams(self):
        """獲取所有活動的串流"""
        return list(self.streams.keys())

    def __del__(self):
        """清理所有串流"""
        for stream_id in list(self.streams.keys()):
            self.remove_stream(stream_id)

def process_rtsp_events(rtsp_manager):
    """處理 RTSP 事件的線程函數"""
    while True:
        try:
            # 從隊列中獲取事件
            event = rtsp_event_queue.get()
            print(f"收到 RTSP 事件: {event}")
            
            if event['type'] == 'add':
                # 添加新的 RTSP 串流
                rtsp_manager.add_stream(event['device_id'], event['rtsp_url'])
                print(f"已添加新的 RTSP 串流: {event['rtsp_url']}")
            elif event['type'] == 'remove':
                # 移除 RTSP 串流
                rtsp_manager.remove_stream(event['device_id'])
                print(f"已移除 RTSP 串流: {event['device_id']}")
                
        except Exception as e:
            print(f"處理 RTSP 事件時發生錯誤: {str(e)}")
            continue
