# YCM 智能聊天系統

基於先進記憶管理的繁體中文聊天機器人，具有語義檢索和長期記憶功能。

## 主要特點

- **繁體中文界面**
  - 完全繁體中文用戶界面
  - 優化的中文語義理解和回應
  - 支持複雜中文表達和語境

- **先進記憶管理**
  - 基於向量的語義記憶存儲
  - 智能記憶檢索和參考
  - 記憶向量化表示和存儲
  - 自動記憶更新和維護

- **模型自動切換機制**
  - 系統實現了智能的模型切換機制，確保服務的穩定性
  - 依據 model_rotation.py 的實現，系統會在多個模型之間進行輪替
  - 每個模型都會被平等地使用，而不是設定主要和備用的模型
  - 完整的錯誤處理和日誌記錄
  - 在聊天歷史中標記使用的模型

- **性能監控與分析**
  - 記錄 API 回應時間
  - 記錄總處理時間
  - 時間戳記錄
  - 錯誤原因記錄

- **數據持久化**
  - 聊天歷史JSON存儲 (test_chat_history.json)
  - 檢索結果緩存 (tmp_retrieval_results.json)
  - 向量化記憶存儲 (vector_store_memories.json)
  - 完整向量表示保存
  - MongoDB 雲端數據存儲

- **測試介面**
  - 提供直觀的測試界面
  - 支持手動測試對話流程
  - 實時顯示測試結果和系統狀態
  - 支持測試數據的導出和分析

- **自動化測試工具**
  - 批量測試對話場景
  - 自動生成測試報告
  - 性能指標統計和分析
  - 支持測試用例的管理和重用

- **資料庫建置**
  - MongoDB Atlas 雲端數據存儲
  - 用戶數據和記憶流同步
  - 數據備份和恢復機制
  - 多用戶數據隔離和管理
  - 支援多執行緒 ID

## 系統架構

### 核心組件

1. **長期記憶代理 (long_term_memory_agent.py)**
   - 管理記憶的儲存和檢索
   - 生成基於記憶的回應
   - 處理記憶參考的格式化
   - 向量化記憶並保存
   - 實現模型切換機制
   - 記錄性能指標

2. **MongoDB 記憶代理 (mongodb_memory_agent.py)**
   - 完全基於 MongoDB 的記憶存儲和檢索
   - 支援多用戶和多執行緒 ID
   - 提供與本地記憶代理相同的功能介面
   - 優化的記憶檢索和更新邏輯

3. **MongoDB 應用入口 (app_mongodb.py)**
   - 提供完全基於 MongoDB 的應用入口
   - 支援命令行參數配置
   - 支援強制重新初始化選項
   - 提供 RESTful API 接口

4. **MongoDB 同步工具 (mongodb_sync.py)**
   - 管理與 MongoDB 的連接
   - 同步本地數據到雲端
   - 從雲端下載數據到本地
   - 處理數據格式轉換和兼容性

5. **LangChain 代理網路介面**
   - 提供 LangChain 代理的 Web 介面
   - 支援多種 LangChain 工具整合
   - 提供 API 接口供外部系統調用
   - 支援自定義代理配置

## 核心功能詳解

### response_generate 函數

`response_generate` 是系統的核心函數，負責生成 AI 回應。

**輸入參數：**
- `user_input` (str): 用戶輸入的文本
- `chat_history` (list): 聊天歷史記錄
- `user_id` (str, 可選): 用戶 ID，默認為 "test_user_1"
- `model` (str, 可選): 使用的模型名稱，默認為 "gpt-4o"
- `temperature` (float, 可選): 模型溫度參數，默認為 0.7
- `max_tokens` (int, 可選): 最大生成令牌數，默認為 1024
- `thread_id` (int, 可選): 執行緒 ID，默認為 1

**輸出：**
- `response` (dict): 包含以下字段的回應字典：
  - `agent_output` (str): AI 生成的回應文本
  - `memory_references` (list): 引用的記憶列表
  - `model_used` (str): 實際使用的模型名稱
  - `timing` (dict): 性能計時信息
  - `error` (str, 可選): 如果發生錯誤，包含錯誤信息

**功能流程：**
1. 檢索相關記憶
2. 構建提示詞和系統消息
3. 嘗試使用主要模型生成回應
4. 如果主要模型失敗，自動切換到備用模型
5. 格式化回應並添加性能指標
6. 保存聊天歷史和更新記憶

### 模型自動切換機制

系統實現了智能的模型切換機制，確保服務的穩定性：

1. **模型輪替**：系統會在多個模型之間進行輪替
2. **錯誤檢測**：監控 API 調用的錯誤和超時
3. **自動切換**：當模型不可用時，自動切換到下一個模型
4. **錯誤記錄**：詳細記錄切換原因和錯誤信息
5. **透明標記**：在回應中明確標記使用的模型
6. **性能監控**：記錄不同模型的回應時間和性能差異

## 資料庫格式

### MongoDB 集合結構

系統使用 MongoDB Atlas 作為雲端數據存儲，主要集合包括：

1. **MEMORY_STREAM_COLLECTION**
   ```json
   {
     "idx": 0,
     "id": "ca7e4e8e-c33a-4dfc-83cf-1aa924cc8f22",
     "user_id": "test_user_1",
     "text": "用戶記憶內容",
     "vector": [0.002, -0.024, 0.031, ...],
     "created_time": "2025-03-11 18:25:26",
     "last_retrieved": "2025-03-11 22:45:12"
   }
   ```

2. **CHAT_HISTORY_COLLECTION**
   ```json
   {
     "user_id": "test_user_1",
     "thread_id": 1,
     "user_input": "用戶輸入的文字",
     "agent_output": "AI 回應的文字",
     "memory_references": [
       {
         "id": "memory_id_1",
         "score": 0.95
       }
     ],
     "model_used": "gpt-4o",
     "timestamp": "2025-03-11 22:45:15",
     "timing": {
       "api_response_time": 2.345,
       "total_processing_time": 3.123
     }
   }
   ```

3. **RETRIEVAL_RESULTS_COLLECTION**
   ```json
   {
     "user_id": "test_user_1",
     "thread_id": 1,
     "timestamp": "2025-03-11 22:45:12",
     "results": [
       {
         "id": "memory_id_1",
         "content": "記憶內容",
         "score": 0.95
       }
     ]
   }
   ```

## app_mongodb.py 參數使用

`app_mongodb.py` 是系統的 MongoDB 版本入口點，支援以下命令行參數：

```bash
python app_mongodb.py [--port PORT] [--host HOST] [--force-reinit] [--debug]
```

- `--port PORT`: 指定服務器端口，默認為 5000
- `--host HOST`: 指定服務器主機地址，默認為 127.0.0.1
- `--force-reinit`: 強制重新初始化 MongoDB 集合（清空現有數據）
- `--debug`: 啟用調試模式，顯示詳細日誌

**使用範例：**

```bash
# 使用默認設置啟動
python app_mongodb.py

# 指定端口和主機
python app_mongodb.py --port 8080 --host 0.0.0.0

# 強制重新初始化數據庫
python app_mongodb.py --force-reinit

# 啟用調試模式
python app_mongodb.py --debug
```

## LangChain 代理網路介面

系統整合了 LangChain 框架，提供強大的代理功能：

1. **Web API 接口**：通過 RESTful API 暴露 LangChain 代理功能
2. **工具整合**：支援多種 LangChain 工具，包括搜索、計算和數據檢索
3. **自定義代理**：支援配置自定義代理行為和工具集
4. **記憶整合**：將 LangChain 代理與系統記憶功能無縫整合
5. **多步驟推理**：支援複雜的多步驟推理任務
6. **結構化輸出**：支援生成結構化的 JSON 輸出

**API 示例：**

```
POST /langchain_agent
{
  "user_input": "幫我查詢今天的天氣",
  "user_id": "test_user_1",
  "tools": ["web_search", "calculator"],
  "thread_id": 1
}
```

## 安裝與使用

### 系統要求

- Python 3.8+
- 網路連接（用於OpenAI API）
- OpenAI API金鑰
- MongoDB Atlas 帳戶（用於雲端數據存儲）

### 安裝步驟

1. 克隆儲存庫：
```bash
git clone [repository-url]
cd YCM-Smart-Access-Control-System
```

2. 創建虛擬環境：
```bash
python -m venv face_recog_env
# Windows
face_recog_env\Scripts\activate
# Linux/Mac
source face_recog_env/bin/activate
```

3. 安裝依賴：
```bash
# 先安裝 PyTorch 相關依賴
pip install -r pytorch_requirements.txt

# 再安裝其他依賴
pip install -r requirements.txt
```

4. 設置環境變數：
創建 `.env` 文件並設置：
```
OPENAI_API_KEY=your_api_key_here
MONGO_URI=your_mongodb_connection_string_here
```

### 啟動 MongoDB 版應用

```bash
python app_mongodb.py
```

### 啟動傳統本地版應用

```bash
python app.py
```

## 開發指南

### 項目結構

```
YCM-Smart-Access-Control-System/

├── utils/
│   ├── mongodb_memory_agent.py   # MongoDB 記憶代理
│   ├── mongodb_sync.py       # MongoDB 同步工具
│   ├── db_init.py            # 資料庫初始化
│   └── ...                   # 其他工具模組
├── memory_store/             # 記憶相關數據
│   ├── chat_history.json     # 聊天歷史
│   ├── tmp_retrieval_results.json # 檢索結果
│   └── memory_stream.json    # 記憶流
└── README.md                 # 本文檔
├── main.py                 # 主程序
├── train_face.py           # 人臉訓練
├── requirements.txt        # 依賴清單
├── services/               # 服務模組
│   ├── face_service.py     # 人臉識別服務
│   ├── llm_service.py      # 語言模型服務
│   ├── common_service.py   # 公共服務
│   └── chatgpt_tts_service.py # 文字轉語音服務
├── utils/                  # 工具模組
│   ├── whisper_speech.py   # 語音識別
│   ├── chat_memory.py      # 對話記憶
│   └── api_cost_tracker.py # API 成本追蹤
├── data/                   # 數據文件
│   ├── face_features.json  # 人臉特徵
│   └── employee_data.json  # 員工資料
├── logs/                   # 日誌文件
│   └── api_cost.json       # API 成本記錄
└── .env                    # 環境設置（不要提交到版本控制）
```

### 主要功能說明

- **記憶存儲**：系統將對話內容自動保存為向量化記憶
- **記憶檢索**：基於語義相似度檢索相關記憶
- **記憶向量化**：將記憶文本轉換為向量表示並保存
- **模型切換**：當模型不可用時自動切換到下一個模型
- **性能監控**：記錄API回應時間和總處理時間
- **聊天歷史格式**：優化的JSON格式，支持多用戶對話存儲
- **去重機制**：防止重複保存相同的對話記錄
- **雲端同步**：自動將本地數據同步到 MongoDB 雲端
- **多用戶支持**：支持多用戶數據隔離和管理

### API請求時間說明

1. **API回應時間 (api_response_time)**
   - 定義：從發送請求到LLM模型直到收到回應的時間
   - 計算方式：`api_response_time = api_call_end_time - api_call_start_time`
   - 包含內容：
     - 網絡傳輸時間
     - 模型處理時間
     - 服務器處理時間

2. **總處理時間 (total_processing_time)**
   - 定義：從用戶發送消息到系統完成所有處理並顯示回應的時間
   - 計算方式：`total_processing_time = process_end_time - process_start_time`
   - 包含內容：
     - API回應時間
     - 記憶檢索時間
     - 記憶向量化時間
     - 本地處理時間

## 未來規劃

### API調用機制

- **引入更多 OpenAI 模型**
  - 基於任務複雜度的智能模型選擇
  - 模型性能比較和評估

### 記憶力壓縮機制

### 記憶力管理系統

### 更新良好測試介面及自動化測試工具

- **增強測試界面**
  - 更直觀的用戶體驗設計
  - 支持更多測試場景和參數配置
  - 實時性能監控和分析
  - 測試數據可視化展示

- **自動化測試工具升級**
  - 支持自定義測試腳本
  - 自動化回歸測試
  - 壓力測試和性能測試
  - 測試覆蓋率分析
  - 測試報告自動生成和發送
