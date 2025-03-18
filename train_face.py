#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
import time
import json
from pathlib import Path
from insightface.app import FaceAnalysis

# 人臉檢測大小
FACE_DETECTION_SIZE = (160, 160)

class FaceTrainer:
    """人臉特徵訓練工具"""
    
    def __init__(self, model_name='buffalo_l'):
        """初始化人臉訓練服務"""
        self.model_name = model_name
        self.providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        
        # 初始化 InsightFace 模型
        self.face_app = FaceAnalysis(
            name=model_name,
            allowed_modules=['detection', 'recognition'],
            providers=self.providers
        )
        
        # 使用較小的檢測尺寸以提高性能
        self.face_app.prepare(
            ctx_id=0 if torch.cuda.is_available() else -1,
            det_size=FACE_DETECTION_SIZE
        )
        
        # 打印 CUDA 狀態
        if torch.cuda.is_available():
            print(f"使用 CUDA 加速，設備: {torch.cuda.get_device_name(0)}")
        else:
            print("無法使用 CUDA 加速，將使用 CPU 模式")
            
        # 確保數據目錄存在
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        self.features_path = os.path.join(self.data_dir, "face_features.json")
        self.training_dir = os.path.join(self.data_dir, "training_images")
        if not os.path.exists(self.training_dir):
            os.makedirs(self.training_dir)
    
    def load_face_features(self):
        """從 JSON 文件加載人臉特徵"""
        if not os.path.exists(self.features_path):
            return {}
            
        try:
            with open(self.features_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"加載人臉特徵時發生錯誤: {e}")
            return {}
    
    def save_face_features(self, face_features):
        """保存人臉特徵到 JSON 文件"""
        try:
            with open(self.features_path, 'w') as f:
                json.dump(face_features, f, indent=2)
            return True
        except Exception as e:
            print(f"保存人臉特徵時發生錯誤: {e}")
            return False
    
    def train_face(self, name=None):
        """訓練人臉特徵
        
        Args:
            name: 人員姓名，如果為 None 則會提示輸入
            
        Returns:
            bool: 是否成功訓練
        """
        if name is None:
            name = input("請輸入人員姓名 (英文或拼音，不含空格): ")
            
        if not name or ' ' in name:
            print("姓名不能為空或包含空格")
            return False
            
        # 創建人員訓練圖像目錄
        person_dir = os.path.join(self.training_dir, name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
            
        # 初始化攝像頭
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("無法開啟攝像頭")
            return False
            
        # 設置攝像頭分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"開始為 {name} 訓練人臉特徵")
        print("請轉動頭部，以不同角度捕獲人臉")
        print("按空格鍵捕獲人臉，按 q 鍵結束訓練")
        
        features = []
        image_count = 0
        
        while image_count < 5:  # 至少捕獲 5 張不同角度的人臉
            ret, frame = cap.read()
            
            if not ret:
                print("無法獲取攝像頭畫面")
                break
                
            # 水平翻轉，使其如鏡子般顯示
            frame = cv2.flip(frame, 1)
            
            # 檢測人臉
            faces = self.face_app.get(frame)
            
            # 顯示指導信息
            cv2.putText(
                frame,
                f"已捕獲: {image_count}/5",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                "按空格鍵捕獲人臉",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # 繪製檢測到的人臉
            if faces:
                for face in faces:
                    bbox = face.bbox.astype(int)
                    cv2.rectangle(
                        frame,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        (0, 255, 0),
                        2
                    )
            
            cv2.imshow("人臉訓練", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # 空格鍵
                if faces:
                    face = faces[0]  # 使用第一個檢測到的人臉
                    
                    # 保存人臉圖像
                    bbox = face.bbox.astype(int)
                    face_image = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    image_path = os.path.join(person_dir, f"{name}_{image_count}.jpg")
                    cv2.imwrite(image_path, face_image)
                    
                    # 獲取人臉特徵
                    feature = face.normed_embedding.tolist()
                    features.append(feature)
                    
                    print(f"捕獲第 {image_count+1} 張人臉圖像")
                    image_count += 1
                    
                    # 等待一段時間，避免連續捕獲相同角度
                    time.sleep(1)
                else:
                    print("未檢測到人臉，請調整位置")
            
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # 保存特徵到 JSON 文件
        try:
            face_features = self.load_face_features()
            
            face_features[name] = features
            
            self.save_face_features(face_features)
            
            print(f"已成功保存 {name} 的 {len(features)} 個人臉特徵")
            return True
        except Exception as e:
            print(f"保存人臉特徵時發生錯誤: {e}")
            return False


def main():
    """主函數"""
    trainer = FaceTrainer()
    trainer.train_face()


if __name__ == "__main__":
    main()
