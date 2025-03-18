"""
MongoDB 同步工具 - 將本地 JSON 檔案同步到 MongoDB 雲端
"""
import json
import os
import sys
import time
from typing import Dict, Any, List, Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from dotenv import load_dotenv

# 添加專案根目錄到路徑，以便正確導入模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mongodb_config import get_mongo_client

# 載入環境變數
load_dotenv()

# MongoDB 連接設定
DB_NAME = "ycm_assistant_db"
CHAT_COLLECTION = "chat_history"  # 修改為與 db_init.py 一致
MEMORY_COLLECTION = "memory_stream"  # 修改為與 db_init.py 一致

# 本地檔案路徑
LOCAL_CHAT_PATH = "memory_store/chat_history.json"
LOCAL_MEMORY_PATH = "memory_store/memory_stream.json"

class MongoDBSync:
    """MongoDB 同步工具類"""
    
    def __init__(self, existing_client=None):
        """初始化 MongoDB 連接
        
        Args:
            existing_client: 可選的現有 MongoDB 客戶端實例
        """
        self.client = None
        self.db = None
        self.chat_collection = None
        self.memory_collection = None
        
        if existing_client:
            # 使用現有的連接
            self.client = existing_client
            self.db = self.client[DB_NAME]
            self.chat_collection = self.db[CHAT_COLLECTION]
            self.memory_collection = self.db[MEMORY_COLLECTION]
            print(f"使用現有連接到 MongoDB: {DB_NAME}")
        else:
            # 創建新連接
            self.connect()
    
    def connect(self) -> bool:
        """連接到 MongoDB"""
        try:
            # 使用修改後的連接配置
            self.client = get_mongo_client()
            if not self.client:
                print("錯誤: 無法創建 MongoDB 客戶端")
                return False
            
            self.db = self.client[DB_NAME]
            self.chat_collection = self.db[CHAT_COLLECTION]
            self.memory_collection = self.db[MEMORY_COLLECTION]
            
            # 測試連接
            self.client.admin.command('ping')
            print(f"成功連接到 MongoDB: {DB_NAME}")
            return True
        except Exception as e:
            print(f"連接 MongoDB 時發生錯誤: {str(e)}")
            return False
    
    def close(self) -> None:
        """關閉 MongoDB 連接"""
        if self.client:
            self.client.close()
            print("已關閉 MongoDB 連接")
    
    def clear_database(self) -> bool:
        """清空資料庫中的所有集合"""
        try:
            if not self.client:
                print("錯誤: 未連接到 MongoDB")
                return False
            
            # 清空聊天歷史集合
            result1 = self.chat_collection.delete_many({})
            print(f"已清空聊天歷史集合，刪除了 {result1.deleted_count} 條記錄")
            
            # 清空記憶流集合
            result2 = self.memory_collection.delete_many({})
            print(f"已清空記憶流集合，刪除了 {result2.deleted_count} 條記錄")
            
            return True
        except Exception as e:
            print(f"清空資料庫時發生錯誤: {str(e)}")
            return False
    
    def upload_chat_history(self, user_id: str = None) -> bool:
        """上傳聊天歷史到 MongoDB"""
        try:
            # 檢查本地檔案是否存在
            if not os.path.exists(LOCAL_CHAT_PATH):
                print(f"錯誤: 找不到本地聊天歷史檔案 {LOCAL_CHAT_PATH}")
                return False
            
            # 檢查連接狀態
            if not self.client:
                print("錯誤: 未連接到 MongoDB")
                return False
            
            # 讀取本地聊天歷史
            with open(LOCAL_CHAT_PATH, "r", encoding="utf-8") as f:
                chat_data = json.load(f)
            
            # 檢查是否有對話需要上傳
            conversations = chat_data.get("conversations", [])
            if not conversations:
                print("沒有聊天歷史需要上傳")
                return True
            
            # 如果指定了用戶 ID，只上傳該用戶的對話
            if user_id:
                conversations = [conv for conv in conversations if conv.get("user_id") == user_id]
                if not conversations:
                    print(f"沒有用戶 {user_id} 的聊天歷史需要上傳")
                    return True
            
            # 為每個對話添加上傳時間
            for conv in conversations:
                conv["uploaded_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                
                # 確保每個對話都有 user_id
                if "user_id" not in conv:
                    conv["user_id"] = "unknown_user"
            
            # 上傳到 MongoDB
            result = self.chat_collection.insert_many(conversations)
            print(f"成功上傳 {len(result.inserted_ids)} 條對話到 MongoDB")
            return True
        except Exception as e:
            print(f"上傳聊天歷史時發生錯誤: {str(e)}")
            return False
    
    def upload_memory_stream(self) -> bool:
        """上傳記憶流到 MongoDB"""
        try:
            # 檢查本地檔案是否存在
            if not os.path.exists(LOCAL_MEMORY_PATH):
                print(f"錯誤: 找不到本地記憶流檔案 {LOCAL_MEMORY_PATH}")
                return False
            
            # 檢查連接狀態
            if not self.client:
                print("錯誤: 未連接到 MongoDB")
                return False
            
            # 讀取本地記憶流
            with open(LOCAL_MEMORY_PATH, "r", encoding="utf-8") as f:
                memory_data = json.load(f)
            
            # 檢查是否有記憶需要上傳
            memories = memory_data.get("memory_stream", [])
            if not memories:
                print("沒有記憶流需要上傳")
                return True
            
            # 為每條記憶添加上傳時間
            for memory in memories:
                memory["uploaded_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                
                # 確保每條記憶都有必要的欄位
                if "user_id" not in memory:
                    memory["user_id"] = "unknown_user"
                if "created_time" not in memory:
                    memory["created_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            
            # 上傳到 MongoDB
            result = self.memory_collection.insert_many(memories)
            print(f"成功上傳 {len(result.inserted_ids)} 條記憶到 MongoDB")
            
            # 更新元數據
            metadata = memory_data.get("metadata", {})
            metadata["uploaded_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            
            # 計算每個用戶的記憶數量
            user_memory_counts = {}
            for memory in memories:
                user_id = memory.get("user_id", "unknown_user")
                if user_id not in user_memory_counts:
                    user_memory_counts[user_id] = 0
                user_memory_counts[user_id] += 1
            
            # 將用戶記憶計數添加到元數據
            if "user_memory_counts" not in metadata:
                metadata["user_memory_counts"] = {}
            
            # 合併現有計數和新計數
            for user_id, count in user_memory_counts.items():
                if user_id in metadata["user_memory_counts"]:
                    metadata["user_memory_counts"][user_id] += count
                else:
                    metadata["user_memory_counts"][user_id] = count
            
            # 上傳元數據到 MongoDB
            metadata_to_upload = metadata.copy()
            self.memory_collection.insert_one({"_type": "metadata", **metadata_to_upload})
            print("成功上傳記憶流元數據")
            
            # 更新本地元數據，確保沒有 ObjectId
            memory_data["metadata"] = self._convert_objectid_to_str(metadata)
            
            # 確保記憶流中沒有 ObjectId
            memory_data["memory_stream"] = self._convert_objectid_to_str(memories)
            
            # 保存到本地
            with open(LOCAL_MEMORY_PATH, "w", encoding="utf-8") as f:
                json.dump(memory_data, f, ensure_ascii=False, indent=2)
            print("已更新本地記憶流元數據")
            
            return True
        except Exception as e:
            print(f"上傳記憶流時發生錯誤: {str(e)}")
            return False
    
    def download_chat_history(self, user_id: str = None) -> bool:
        """從 MongoDB 下載聊天歷史到本地"""
        try:
            # 準備查詢條件
            query = {"user_id": user_id} if user_id else {}
            
            # 從 MongoDB 獲取對話
            conversations = list(self.chat_collection.find(query, {"_id": 0}))
            
            # 確保 memory_store 資料夾存在
            os.makedirs("memory_store", exist_ok=True)
            
            # 構建聊天歷史資料
            chat_data = {"conversations": conversations}
            
            # 保存到本地
            with open(LOCAL_CHAT_PATH, "w", encoding="utf-8") as f:
                json.dump(chat_data, f, ensure_ascii=False, indent=2)
            
            print(f"成功下載 {len(conversations)} 條對話到本地")
            return True
        except Exception as e:
            print(f"下載聊天歷史時發生錯誤: {str(e)}")
            return False
    
    def download_memory_stream(self) -> bool:
        """從 MongoDB 下載記憶流到本地"""
        try:
            # 獲取記憶流
            memories = list(self.memory_collection.find({"_type": {"$ne": "metadata"}}, {"_id": 0}))
            
            # 獲取元數據
            metadata_doc = self.memory_collection.find_one({"_type": "metadata"}, {"_id": 0})
            if not metadata_doc:
                metadata_doc = {"memory_count": len(memories), "last_updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
            
            # 移除 _type 字段
            if "_type" in metadata_doc:
                del metadata_doc["_type"]
            
            # 處理 ObjectId
            metadata_doc = self._convert_objectid_to_str(metadata_doc)
            memories = self._convert_objectid_to_str(memories)
            
            # 構建記憶流資料
            memory_data = {
                "metadata": metadata_doc,
                "memory_stream": memories
            }
            
            # 確保 memory_store 資料夾存在
            os.makedirs("memory_store", exist_ok=True)
            
            # 保存到本地
            with open(LOCAL_MEMORY_PATH, "w", encoding="utf-8") as f:
                json.dump(memory_data, f, ensure_ascii=False, indent=2)
            
            print(f"成功下載 {len(memories)} 條記憶到本地")
            return True
        except Exception as e:
            print(f"下載記憶流時發生錯誤: {str(e)}")
            return False
    
    def _convert_objectid_to_str(self, data):
        """將 ObjectId 轉換為字符串，以便 JSON 序列化
        
        Args:
            data: 需要轉換的數據，可以是字典、列表或其他類型
            
        Returns:
            轉換後的數據
        """
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if key == "_id" and hasattr(value, "__str__"):
                    # 將 ObjectId 轉換為字符串
                    result[key] = str(value)
                else:
                    # 遞歸處理嵌套的字典和列表
                    result[key] = self._convert_objectid_to_str(value)
            return result
        elif isinstance(data, list):
            return [self._convert_objectid_to_str(item) for item in data]
        elif hasattr(data, "__str__") and type(data).__name__ == "ObjectId":
            # 將 ObjectId 轉換為字符串
            return str(data)
        else:
            # 其他類型直接返回
            return data
    
    def sync_all(self, user_id: str = None, clear_first: bool = False) -> bool:
        """同步所有資料"""
        if clear_first:
            self.clear_database()
        
        upload_chat = self.upload_chat_history(user_id)
        upload_memory = self.upload_memory_stream()
        return upload_chat and upload_memory
    
    def download_all(self, user_id: str = None) -> bool:
        """下載所有資料"""
        download_chat = self.download_chat_history(user_id)
        download_memory = self.download_memory_stream()
        return download_chat and download_memory


# 直接執行時的範例用法
if __name__ == "__main__":
    import argparse
    
    # 解析命令行參數
    parser = argparse.ArgumentParser(description="MongoDB 同步工具")
    parser.add_argument("--action", choices=["upload", "download", "sync", "clear"], default="sync", help="要執行的操作")
    parser.add_argument("--user", help="指定用戶 ID")
    parser.add_argument("--clear", action="store_true", help="在上傳前清空資料庫")
    args = parser.parse_args()
    
    # 創建同步工具實例
    sync_tool = MongoDBSync()
    
    try:
        if args.action == "clear":
            sync_tool.clear_database()
        elif args.action == "upload":
            if args.clear:
                sync_tool.clear_database()
            sync_tool.upload_chat_history(args.user)
            sync_tool.upload_memory_stream()
        elif args.action == "download":
            sync_tool.download_chat_history(args.user)
            sync_tool.download_memory_stream()
        elif args.action == "sync":
            # 先上傳再下載
            sync_tool.sync_all(args.user, args.clear)
    finally:
        sync_tool.close()
