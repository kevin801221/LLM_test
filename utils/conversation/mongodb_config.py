"""
MongoDB 配置工具 - 提供更安全的連接選項
"""
import os
from pymongo import MongoClient
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 獲取原始連接字串
ORIGINAL_MONGO_URI = os.getenv("MONGO_URI", "")

# 修改後的連接選項
def get_mongo_client():
    """
    獲取配置了適當選項的 MongoDB 客戶端
    
    返回:
        MongoClient: 配置好的 MongoDB 客戶端
    """
    try:
        # 檢查連接字串是否為空
        uri = ORIGINAL_MONGO_URI
        if not uri:
            print("錯誤: MONGO_URI 環境變數未設置或為空")
            return None
            
        # 使用原始連接字串，不添加額外參數
        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=10000,    # 延長超時時間
            connectTimeoutMS=20000,            # 延長連接超時
            socketTimeoutMS=20000              # 延長 Socket 超時
        )
        return client
    except Exception as e:
        print(f"創建 MongoDB 客戶端時發生錯誤: {str(e)}")
        return None

# 測試連接
def test_connection():
    """
    測試 MongoDB 連接
    
    返回:
        bool: 連接是否成功
    """
    client = get_mongo_client()
    if not client:
        return False
    
    try:
        # 測試連接
        client.admin.command('ping')
        print("MongoDB 連接測試成功!")
        return True
    except Exception as e:
        print(f"MongoDB 連接測試失敗: {str(e)}")
        return False
    finally:
        if client:
            client.close()

if __name__ == "__main__":
    # 執行連接測試
    test_connection()
