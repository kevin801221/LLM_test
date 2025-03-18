"""
模型輪換機制
提供同一個模型但使用不同 API key 的輪換和回退功能
"""

import time
import os
from typing import Any, Tuple, Dict, List
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 加載 .env 文件
load_dotenv()

# 默認模型
DEFAULT_MODEL = "gpt-4o"

# 可用模型列表（為了向後兼容性）
AVAILABLE_MODELS = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]

# 定義 API keys 列表和對應的環境變數名稱
API_KEYS = []
API_KEY_NAMES = []

# 從 .env 文件讀取 API keys
def load_api_keys_from_env():
    """從 .env 文件讀取 API keys"""
    global API_KEYS, API_KEY_NAMES
    API_KEYS = []
    API_KEY_NAMES = []
    
    # 從環境變數讀取 API keys
    i = 1
    while True:
        key_name = f"OPENAI_API_KEY_{i}"
        if key_name in os.environ and os.environ[key_name]:
            API_KEYS.append(os.environ[key_name])
            API_KEY_NAMES.append(key_name)
            i += 1
        else:
            # 如果找不到更多的 key，檢查是否有默認的 OPENAI_API_KEY
            if i == 1 and "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]:
                API_KEYS.append(os.environ["OPENAI_API_KEY"])
                API_KEY_NAMES.append("OPENAI_API_KEY")
            break
    
    return len(API_KEYS) > 0

# 記錄當前使用的 API key 索引
current_key_index = 0

# 為了向後兼容性，提供 current_model_index 作為別名
current_model_index = current_key_index

# 記錄 API key 使用情況
api_key_usage_stats = {}

def initialize_rotation(model_name=DEFAULT_MODEL):
    """
    初始化模型輪換系統
    
    Args:
        model_name: 要使用的模型名稱
    
    Returns:
        是否成功初始化
    """
    global api_key_usage_stats
    
    # 加載 API keys
    if not load_api_keys_from_env():
        print("錯誤: 未找到任何 API keys。請在 .env 文件中設置 OPENAI_API_KEY 或 OPENAI_API_KEY_1, OPENAI_API_KEY_2, ... 環境變數")
        return False
    
    # 初始化使用統計
    api_key_usage_stats = {
        key_name: {
            "key_id": API_KEYS[i][-8:], 
            "success": 0, 
            "failure": 0, 
            "last_used": None, 
            "model": model_name
        } 
        for i, key_name in enumerate(API_KEY_NAMES)
    }
    
    print(f"模型輪換系統已初始化，使用模型: {model_name}，可用 API keys:")
    for i, name in enumerate(API_KEY_NAMES):
        print(f"  - {name}")
    return True

def get_model(model_name=None, force_model=False) -> Tuple[Any, str]:
    """
    獲取模型實例，支持 API key 輪換機制
    
    Args:
        model_name: 指定要使用的模型名稱，如果為None則使用默認模型
        force_model: 是否強制使用指定的模型
    
    Returns:
        模型實例和使用的模型名稱的元組，如果無法創建模型則返回 (None, None)
    """
    global current_key_index, api_key_usage_stats
    
    # 如果 API keys 列表為空，嘗試加載
    if not API_KEYS and not load_api_keys_from_env():
        print("錯誤: 未找到任何 API keys。請在 .env 文件中設置 OPENAI_API_KEY 或 OPENAI_API_KEY_1, OPENAI_API_KEY_2, ... 環境變數")
        return None, None
    
    # 使用指定的模型名稱或默認模型
    model_to_use = model_name if model_name is not None else DEFAULT_MODEL
    
    # 保存當前 API key 索引，用於輪換
    start_index = current_key_index
    model = None
    current_key = None
    key_id = None
    key_name = None
    
    # 嘗試所有 API keys，直到找到一個可用的
    for _ in range(len(API_KEYS)):
        current_key = API_KEYS[current_key_index]
        key_name = API_KEY_NAMES[current_key_index]
        key_id = current_key[-8:]  # 使用 API key 的最後 8 位作為標識
        
        try:
            # 使用 tiktoken_model 參數解決編碼器問題
            model = ChatOpenAI(
                model_name=model_to_use, 
                openai_api_key=current_key,
                tiktoken_model_name="gpt-4"  # 使用固定的 tiktoken 模型名稱
            )
            
            # 更新使用統計
            if key_name not in api_key_usage_stats:
                api_key_usage_stats[key_name] = {
                    "key_id": key_id,
                    "success": 0, 
                    "failure": 0, 
                    "last_used": None, 
                    "model": model_to_use
                }
            
            api_key_usage_stats[key_name]["success"] += 1
            api_key_usage_stats[key_name]["last_used"] = time.strftime("%Y-%m-%d %H:%M:%S")
            api_key_usage_stats[key_name]["model"] = model_to_use
            
            print(f"使用模型: {model_to_use}, API key: {key_name}")
            
            # 成功創建模型後，移動到下一個 API key 以備下次使用
            current_key_index = (current_key_index + 1) % len(API_KEYS)
            break
        except Exception as e:
            print(f"使用 API key {key_name} 創建 {model_to_use} 模型實例時發生錯誤: {str(e)}")
            
            # 更新使用統計
            if key_name not in api_key_usage_stats:
                api_key_usage_stats[key_name] = {
                    "key_id": key_id,
                    "success": 0, 
                    "failure": 0, 
                    "last_used": None, 
                    "model": model_to_use
                }
            
            api_key_usage_stats[key_name]["failure"] += 1
            
            # 嘗試下一個 API key
            current_key_index = (current_key_index + 1) % len(API_KEYS)
    
    # 如果嘗試了所有 API keys 但都失敗了，重置索引
    if model is None:
        current_key_index = 0
        return None, None
    
    return model, model_to_use  # 返回模型名稱而不是 API key 名稱

def get_api_key_usage_stats() -> Dict[str, Dict]:
    """
    獲取 API key 使用統計信息
    
    Returns:
        包含各個 API key 使用情況的字典
    """
    return api_key_usage_stats

def get_model_usage_stats() -> Dict[str, Dict]:
    """
    獲取 API key 使用統計信息 (向後兼容函數)
    
    Returns:
        包含各個 API key 使用情況的字典
    """
    return get_api_key_usage_stats()

def reset_key_rotation():
    """
    重置 API key 輪換狀態
    """
    global current_key_index
    current_key_index = 0
    print("API key 輪換狀態已重置")

def reset_model_rotation():
    """
    重置模型輪換狀態 (向後兼容函數)
    """
    return reset_key_rotation()

def set_default_model(model_name):
    """
    設置默認模型
    
    Args:
        model_name: 新的默認模型名稱
    """
    global DEFAULT_MODEL
    DEFAULT_MODEL = model_name
    print(f"默認模型已設置為: {DEFAULT_MODEL}")

def add_api_key(api_key, key_name=None):
    """
    添加新的 API key 到輪換系統
    
    Args:
        api_key: 要添加的 API key
        key_name: API key 的名稱，如果為 None，則自動生成
    """
    global API_KEYS, API_KEY_NAMES
    if api_key not in API_KEYS:
        API_KEYS.append(api_key)
        
        # 如果沒有提供名稱，自動生成一個
        if key_name is None:
            key_name = f"OPENAI_API_KEY_{len(API_KEYS)}"
        
        API_KEY_NAMES.append(key_name)
        key_id = api_key[-8:]
        
        api_key_usage_stats[key_name] = {
            "key_id": key_id,
            "success": 0, 
            "failure": 0, 
            "last_used": None, 
            "model": DEFAULT_MODEL
        }
        print(f"已添加新的 API key: {key_name}")
    else:
        print("API key 已存在")

def remove_api_key(api_key):
    """
    從輪換系統中移除 API key
    
    Args:
        api_key: 要移除的 API key
    """
    global API_KEYS, API_KEY_NAMES, current_key_index
    if api_key in API_KEYS:
        index = API_KEYS.index(api_key)
        key_name = API_KEY_NAMES[index]
        key_id = api_key[-8:]
        
        API_KEYS.pop(index)
        API_KEY_NAMES.pop(index)
        
        # 如果移除的是當前索引或之前的 key，調整當前索引
        if index <= current_key_index and current_key_index > 0:
            current_key_index -= 1
        
        # 如果移除後列表為空，重置索引
        if not API_KEYS:
            current_key_index = 0
        
        print(f"已移除 API key: {key_name}")
    else:
        print("API key 不存在")

def print_api_key_status():
    """
    打印所有 API key 的當前狀態
    """
    print("\n當前 API key 狀態:")
    print("-" * 50)
    print(f"{'API Key 名稱':<20} {'成功次數':<10} {'失敗次數':<10} {'最後使用時間':<20}")
    print("-" * 50)
    
    for key_name, stats in api_key_usage_stats.items():
        print(f"{key_name:<20} {stats['success']:<10} {stats['failure']:<10} {stats['last_used'] or 'N/A':<20}")
    
    print("-" * 50)
    print(f"當前使用索引: {current_key_index} ({API_KEY_NAMES[current_key_index] if API_KEY_NAMES else 'N/A'})")
    print(f"默認模型: {DEFAULT_MODEL}")
    print("-" * 50)

# 初始化時自動加載 API keys
load_api_keys_from_env()

# 主程序入口，用於直接測試
if __name__ == "__main__":
    import time
    
    print("=" * 50)
    print("API Key 輪換機制測試")
    print("=" * 50)
    
    # 初始化輪換系統
    print("\n1. 初始化輪換系統")
    initialize_rotation()
    
    # 顯示當前狀態
    print("\n2. 初始狀態")
    print_api_key_status()
    
    # 模擬使用模型
    print("\n3. 模擬使用模型 (3次)")
    for i in range(3):
        print(f"\n第 {i+1} 次調用:")
        model, model_info = get_model()
        if model:
            print(f"成功獲取模型，使用: {model_info}")
        else:
            print("無法獲取模型")
        time.sleep(1)  # 稍微暫停以便觀察
    
    # 再次顯示狀態
    print("\n4. 使用後狀態")
    print_api_key_status()
    
    # 測試添加新的 API key
    print("\n5. 測試添加新的 API key")
    add_api_key("sk-test-key", "TEST_API_KEY")
    print_api_key_status()
    
    # 測試移除 API key
    print("\n6. 測試移除 API key")
    if API_KEYS:
        remove_api_key(API_KEYS[0])
    print_api_key_status()
    
    # 測試重置輪換
    print("\n7. 測試重置輪換")
    reset_key_rotation()
    print(f"當前索引已重置為: {current_key_index}")
    
    # 測試設置默認模型
    print("\n8. 測試設置默認模型")
    original_model = DEFAULT_MODEL
    set_default_model("gpt-4-turbo")
    print(f"模型已從 {original_model} 更改為 {DEFAULT_MODEL}")
    
    print("\n" + "=" * 50)
    print("測試完成")
    print("=" * 50)
