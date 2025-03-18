"""
MongoDB 初始化工具 - 在啟動介面時清空並初始化 MongoDB 資料庫
"""
import os
import sys
import json
from dotenv import load_dotenv

# 添加專案根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 導入 MongoDB 配置
from utils.mongodb_config import get_mongo_client

# 載入環境變數
load_dotenv()

# MongoDB 連接設定
DB_NAME = 'ycm_assistant_db'

# 集合名稱
CHAT_HISTORY_COLLECTION = 'chat_history'
MEMORY_STREAM_COLLECTION = 'memory_stream'
RETRIEVAL_COLLECTION = 'retrieval_results'

def initialize_database():
    """初始化資料庫，確保集合存在並清空，返回連接實例"""
    print("正在連接 MongoDB...")
    
    try:
        # 使用配置工具獲取 MongoDB 客戶端
        client = get_mongo_client()
        if not client:
            print("無法獲取 MongoDB 客戶端")
            return None
            
        print("連接成功")
        
        # 列出所有資料庫
        all_dbs = client.list_database_names()
        print(f"所有可用資料庫: {all_dbs}")
        
        # 選擇或創建資料庫
        db = client[DB_NAME]
        print(f"已選擇資料庫: {DB_NAME}")
        
        # 列出資料庫中的集合
        collections = db.list_collection_names()
        print(f"資料庫中的集合: {collections}")
        
        # 確保必要的集合存在
        if CHAT_HISTORY_COLLECTION not in collections:
            print(f"創建集合: {CHAT_HISTORY_COLLECTION}")
            db.create_collection(CHAT_HISTORY_COLLECTION)
        
        if MEMORY_STREAM_COLLECTION not in collections:
            print(f"創建集合: {MEMORY_STREAM_COLLECTION}")
            db.create_collection(MEMORY_STREAM_COLLECTION)
            
        if RETRIEVAL_COLLECTION not in collections:
            print(f"創建集合: {RETRIEVAL_COLLECTION}")
            db.create_collection(RETRIEVAL_COLLECTION)
        
        # 清空集合
        print("正在清空集合...")
        chat_result = db[CHAT_HISTORY_COLLECTION].delete_many({})
        memory_result = db[MEMORY_STREAM_COLLECTION].delete_many({})
        retrieval_result = db[RETRIEVAL_COLLECTION].delete_many({})
        
        print(f"已清空聊天歷史集合，刪除了 {chat_result.deleted_count} 條記錄")
        print(f"已清空記憶流集合，刪除了 {memory_result.deleted_count} 條記錄")
        print(f"已清空檢索結果集合，刪除了 {retrieval_result.deleted_count} 條記錄")
        
        print("資料庫已成功清空，準備好接收新的資料")
        
        # 返回連接實例，而不是關閉它
        return client
    except Exception as e:
        print(f"初始化資料庫時發生錯誤: {str(e)}")
        return None

if __name__ == "__main__":
    # 執行初始化
    client = initialize_database()
    if client:
        client.close()
        print("已關閉 MongoDB 連接")
