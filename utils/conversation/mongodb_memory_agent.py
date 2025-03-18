"""
MongoDB 記憶代理 - 完全使用 MongoDB 存儲和檢索記憶，不依賴本地文件
"""
import os
import uuid
import time
from typing import List, Dict, Any, Optional, Literal
from dotenv import load_dotenv
load_dotenv()

import tiktoken
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage, get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from bson.objectid import ObjectId
import logging

# 導入模型輪換模組
from utils.conversation.model_rotation import get_model, get_model_usage_stats, reset_model_rotation
# 導入 MongoDB 相關模組
from utils.conversation.mongodb_config import get_mongo_client

# 從 model_rotation 獲取 API key
from utils.conversation.model_rotation import API_KEYS, load_api_keys_from_env

# 確保 API keys 已加載
if not API_KEYS:
    load_api_keys_from_env()

# 使用第一個可用的 API key 初始化 OpenAIEmbeddings
api_key = API_KEYS[0] if API_KEYS else os.environ.get("OPENAI_API_KEY", "")

# MongoDB 連接設定
DB_NAME = "ycm_assistant_db"
CHAT_COLLECTION = "chat_history"
MEMORY_STREAM_COLLECTION = "memory_stream"
RETRIEVAL_COLLECTION = "retrieval_results"

# 使用第一個可用的 API key 初始化 OpenAIEmbeddings
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# 初始化向量存儲用於記憶
recall_vector_store = InMemoryVectorStore(embeddings)

# 定義用於獲取用戶ID的函數
def get_user_id(config: RunnableConfig) -> str:
    """從 RunnableConfig 中獲取用戶 ID
    
    Args:
        config: RunnableConfig 對象
        
    Returns:
        用戶 ID 字符串
        
    Raises:
        ValueError: 如果未提供用戶 ID
    """
    # 嘗試從不同的可能位置獲取用戶 ID
    user_id = None
    
    # 新版本 LangChain API
    if hasattr(config, "user_id"):
        user_id = config.user_id
    # 舊版本 LangChain API
    elif "configurable" in config and isinstance(config["configurable"], dict):
        user_id = config["configurable"].get("user_id")
    # 直接從字典獲取
    elif isinstance(config, dict):
        user_id = config.get("user_id")
    
    if user_id is None:
        logging.warning("未提供用戶 ID，使用默認值 'default_user'")
        return "default_user"
    
    return user_id

# 創建一個字典來存儲記憶的最後檢索時間
memory_last_retrieved = {}

# 定義記憶工具
@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """Save memory to MongoDB for later semantic retrieval."""
    user_id = get_user_id(config)
    current_time = time.time()
    current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
    memory_id = str(uuid.uuid4())
    
    # 創建文檔
    document = Document(
        page_content=memory,
        metadata={
            "user_id": user_id, 
            "timestamp": current_time,
            "id": memory_id,
            "created_time": current_time_str,
            "last_retrieved": "從未檢索"
        },
    )
    document.id = memory_id
    
    # 添加到向量存儲
    recall_vector_store.add_documents([document])
    
    # 直接將記憶保存到 MongoDB，而不是通過 export_vector_store_memories_to_mongodb
    try:
        # 為文檔獲取向量
        embedding_model = OpenAIEmbeddings(openai_api_key=API_KEYS[0] if API_KEYS else os.environ.get("OPENAI_API_KEY", ""))
        vector = embedding_model.embed_query(memory)
        
        # 獲取 MongoDB 客戶端
        mongo_client = get_mongo_client()
        if mongo_client is None:
            logging.error("無法獲取 MongoDB 客戶端，記憶將只存在於內存中")
            return memory
            
        db = mongo_client[DB_NAME]
        memory_collection = db[MEMORY_STREAM_COLLECTION]
        
        # 創建符合新格式的記憶數據
        memory_data = {
            "idx": 0,  # 這個索引在批量導入時會被更新
            "id": memory_id,
            "user_id": user_id,
            "text": memory,
            "content": memory,  # 添加 content 欄位，與 text 相同
            "vector": [float(v) for v in vector],  # 確保向量中的所有值都是 float 類型
            "created_time": current_time_str,
            "last_retrieved": "從未檢索"
        }
        
        # 插入記憶到 MongoDB
        memory_collection.insert_one(memory_data)
        logging.info(f"已直接保存新記憶到 MongoDB: ID={memory_id}, 用戶ID={user_id}")
    except Exception as e:
        logging.error(f"直接保存記憶到 MongoDB 時出錯: {e}")
        logging.error("記憶將只存在於內存中")
    
    return memory

@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[Dict[str, Any]]:
    """Search for relevant memories in MongoDB."""
    user_id = get_user_id(config)

    def _filter_function(doc: Document) -> bool:
        return doc.metadata.get("user_id") == user_id

    # 使用 similarity_search_with_score 獲取文檔和分數
    documents_with_scores = recall_vector_store.similarity_search_with_score(
        query, k=3, filter=_filter_function
    )
    
    # 更新記憶的最後檢索時間
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    # 獲取 MongoDB 客戶端
    mongo_client = get_mongo_client()
    
    # 準備檢索結果
    retrieval_results = []
    
    for document, score in documents_with_scores:
        memory_id = getattr(document, "id", None)
        if not memory_id or memory_id == "unknown":
            memory_id = document.metadata.get("id", str(uuid.uuid4()))
        
        # 更新最後檢索時間（僅在內存中）
        memory_last_retrieved[memory_id] = current_time
        
        # 如果連接到 MongoDB，僅更新記憶的最後檢索時間，不創建新記錄
        if mongo_client is not None:
            try:
                db = mongo_client[DB_NAME]
                memory_collection = db[MEMORY_STREAM_COLLECTION]  # 使用正確的集合名稱
                
                # 檢查記憶是否存在
                existing_memory = memory_collection.find_one({"id": memory_id})
                
                if existing_memory:
                    # 僅更新最後檢索時間，不修改其他欄位
                    update_result = memory_collection.update_one(
                        {"id": memory_id},
                        {"$set": {"last_retrieved": current_time}}
                    )
                    
                    if update_result.matched_count == 0:
                        logging.warning(f"未找到ID為 {memory_id} 的記憶進行更新")
                else:
                    logging.warning(f"記憶 {memory_id} 不存在於 MongoDB 中，無法更新檢索時間")
            except Exception as e:
                logging.error(f"更新記憶檢索時間時發生錯誤: {str(e)}")
        
        # 添加到檢索結果
        result = {
            "id": memory_id,
            "content": document.page_content,
            "text": document.page_content,  # 添加 text 欄位
            "score": score
        }
        retrieval_results.append(result)
    
    # 將檢索結果保存到 MongoDB（僅作為臨時結果，不影響主記憶庫）
    if mongo_client is not None and retrieval_results:
        try:
            db = mongo_client[DB_NAME]
            retrieval_collection = db[RETRIEVAL_COLLECTION]
            
            # 清空現有檢索結果
            retrieval_collection.delete_many({"user_id": user_id})
            
            # 添加用戶ID和時間戳
            for result in retrieval_results:
                result["user_id"] = user_id
                result["timestamp"] = current_time
            
            # 寫入新檢索結果
            retrieval_collection.insert_many(retrieval_results)
        except Exception as e:
            logging.error(f"保存檢索結果到 MongoDB 時發生錯誤: {str(e)}")
    
    # 返回記憶的 ID、內容和相似度分數
    return retrieval_results

# 初始化搜索工具
search = TavilySearchResults(max_results=1)
tools = [save_recall_memory, search_recall_memories, search]

# 定義狀態類型
class State(MessagesState):
    # 添加基於對話上下文檢索的記憶
    recall_memories: List[str]

# 定義代理提示模板
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一個具有進階長期記憶能力的繁體中文助手 你叫做艾瑪，英文Emma。作為一個無狀態的語言模型，"
            "你必須依靠外部記憶來在對話之間存儲信息。"
            "請利用可用的記憶工具來存儲和檢索重要細節，這將幫助你更好地滿足用戶的需求"
            "並理解他們的上下文。\n\n"
            "記憶使用指南：\n"
            "1. 積極使用記憶工具(save_recall_memory)來建立對用戶的全面理解。\n"
            "2. 根據存儲的記憶做出明智的推測和推斷。\n"
            "3. 定期反思過去的互動以識別模式和偏好。\n"
            "4. 隨著每一條新信息更新你對用戶的心智模型。\n"
            "5. 將新信息與現有記憶交叉參考以保持一致性。\n"
            "6. 優先存儲情感上下文和個人價值觀以及事實。\n"
            "7. 使用記憶來預測需求並根據用戶的風格定制回應。\n"
            "8. 識別並承認用戶情況或觀點隨時間的變化。\n"
            "9. 利用記憶提供個性化的例子和類比。\n"
            "10. 回顧過去的挑戰或成功以指導當前的問題解決。\n\n"
            "## 回憶記憶\n"
            "回憶記憶是根據當前對話上下文檢索的：\n{recall_memories}\n\n"
            "## 指示\n"
            "自然地與用戶互動，就像一個值得信賴的同事或朋友。"
            "無需明確提及你的記憶能力。相反，將你對用戶的理解無縫地融入你的回應中。"
            "注意微妙的線索和潛在的情緒。根據用戶的偏好和當前情緒狀態調整你的溝通風格。"
            "使用工具來持久化你想在下一次對話中保留的信息。"
            "如果你確實調用了工具，所有在工具調用之前的文本都是內部消息。"
            "在調用工具後回應，一旦你確認工具成功完成。\n\n",
        ),
        ("placeholder", "{messages}"),
    ]
)

# 嘗試初始化主模型
try:
    # 使用模型輪換機制獲取模型
    model, current_model_name = get_model()
    if model is None:
        raise ValueError("無法初始化模型，所有 API keys 都失敗")
    
    model_with_tools = model.bind_tools(tools)
    
    # 使用固定的 tiktoken 模型名稱，避免使用 API key 名稱
    try:
        tokenizer = tiktoken.encoding_for_model("gpt-4")
    except:
        # 如果 gpt-4 不可用，嘗試使用 gpt-3.5-turbo
        tokenizer = tiktoken.get_encoding("cl100k_base")  # 這是 GPT-4 和 GPT-3.5 使用的編碼器
except Exception as e:
    logging.error(f"初始化模型時發生錯誤: {str(e)}")
    
    # 嘗試從環境變數直接獲取 API key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and API_KEYS and len(API_KEYS) > 0:
        api_key = API_KEYS[0]  # 使用第一個可用的 API key
    
    if api_key:
        # 如果所有模型都失敗，使用一個基本的回退模型
        model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)
        model_with_tools = model.bind_tools(tools)
        current_model_name = "gpt-3.5-turbo"
        tokenizer = tiktoken.get_encoding("cl100k_base")  # 使用 GPT-3.5 的編碼器
    else:
        logging.error("錯誤: 無法初始化模型，沒有可用的 API key")
        model = None
        model_with_tools = None
        current_model_name = None
        tokenizer = None

# 定義代理節點
def agent(state: State) -> State:
    """Process the current state and generate a response using the LLM."""
    bound = prompt | model_with_tools
    recall_str = (
        "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
    )
    prediction = bound.invoke(
        {
            "messages": state["messages"],
            "recall_memories": recall_str,
        }
    )
    return {
        "messages": [prediction],
    }

# 定義記憶加載節點
def load_memories(state: State, config: RunnableConfig) -> State:
    """Load memories for the current conversation."""
    convo_str = get_buffer_string(state["messages"])
    convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])
    recall_memories = search_recall_memories.invoke(convo_str, config)
    return {
        "recall_memories": [f"{memory['id']}: {memory['content']}" for memory in recall_memories],
    }

# 定義路由工具節點
def route_tools(state: State):
    """Determine whether to use tools or end the conversation based on the last message."""
    msg = state["messages"][-1]
    if msg.tool_calls:
        return "tools"
    return END

# 創建圖
def create_memory_graph():
    """Create the memory graph."""
    # 創建圖
    builder = StateGraph(State)
    
    # 添加節點
    builder.add_node(load_memories)
    builder.add_node(agent)
    builder.add_node("tools", ToolNode(tools))
    
    # 添加邊
    builder.add_edge(START, "load_memories")
    builder.add_edge("load_memories", "agent")
    builder.add_conditional_edges("agent", route_tools, ["tools", END])
    builder.add_edge("tools", "agent")
    
    # 編譯圖
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

# 將聊天歷史保存到 MongoDB
def save_chat_history_to_mongodb(user_id: str, messages: List[Dict[str, Any]]) -> None:
    """將聊天歷史保存到 MongoDB"""
    try:
        # 獲取 MongoDB 客戶端
        mongo_client = get_mongo_client()
        if mongo_client is None:
            logging.error("無法獲取 MongoDB 客戶端")
            return
        
        db = mongo_client[DB_NAME]
        chat_collection = db[CHAT_COLLECTION]
        
        # 處理消息並添加到對話記錄中
        new_conversations = []
        for i in range(0, len(messages), 2):
            if i+1 < len(messages):  # 確保有成對的消息（用戶+AI）
                user_msg = messages[i]
                ai_msg = messages[i+1]
                
                # 創建對話記錄
                timestamp = ai_msg.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                conversation = {
                    "user_id": user_id,
                    "user_input": user_msg.get("content", ""),
                    "agent_output": ai_msg.get("content", ""),
                    "memory_references": ai_msg.get("memory_references", []),
                    "model_used": ai_msg.get("model_used", "unknown"),
                    "timestamp": timestamp,
                    "timing": ai_msg.get("timing", {}),
                    "created_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "thread_id": user_msg.get("thread_id", 1)  # 添加 thread_id 欄位
                }
                
                # 添加到新對話列表
                new_conversations.append(conversation)
        
        # 將新對話寫入 MongoDB
        if new_conversations:
            chat_collection.insert_many(new_conversations)
            logging.info(f"已將 {len(new_conversations)} 條新對話寫入 MongoDB")
    except Exception as e:
        logging.error(f"保存聊天歷史到 MongoDB 時發生錯誤: {str(e)}")

# 將向量存儲記憶導出到 MongoDB
def export_vector_store_memories_to_mongodb() -> None:
    """將向量存儲記憶導出到 MongoDB"""
    try:
        # 使用 similarity_search_with_score 方法獲取所有文檔
        # 使用一個通用查詢來獲取所有記憶
        try:
            retrieved_docs_with_scores = recall_vector_store.similarity_search_with_score(
                "all memories", k=100  # 嘗試獲取最多100條記憶
            )
            docs = [doc for doc, _ in retrieved_docs_with_scores]
        except Exception as e:
            logging.error(f"使用 similarity_search_with_score 獲取記憶時出錯: {e}")
            # 嘗試使用 similarity_search 方法
            try:
                docs = recall_vector_store.similarity_search(
                    "all memories", k=100  # 嘗試獲取最多100條記憶
                )
            except Exception as e2:
                logging.error(f"使用 similarity_search 獲取記憶時出錯: {e2}")
                docs = []
        
        if not docs:
            logging.info("向量存儲中沒有記憶")
            return
            
        logging.info(f"從向量存儲中獲取到 {len(docs)} 條記憶")
        
        # 為每個文檔獲取向量
        embedding_model = OpenAIEmbeddings(openai_api_key=API_KEYS[0] if API_KEYS else os.environ.get("OPENAI_API_KEY", ""))
        
        # 獲取 MongoDB 客戶端
        mongo_client = get_mongo_client()
        if mongo_client is None:
            logging.error("無法獲取 MongoDB 客戶端")
            return
            
        db = mongo_client[DB_NAME]
        memory_collection = db[MEMORY_STREAM_COLLECTION]
        
        # 準備新記憶
        new_memories = []
        
        for idx, doc in enumerate(docs):
            try:
                # 使用 OpenAI 嵌入模型獲取向量
                vector = embedding_model.embed_query(doc.page_content)
                
                # 獲取記憶 ID，優先使用 doc.id，如果不存在則從 metadata 中獲取或生成新的 UUID
                memory_id = getattr(doc, "id", None)
                if not memory_id or memory_id == "unknown":
                    memory_id = doc.metadata.get("id", str(uuid.uuid4()))
                
                # 獲取最後檢索時間
                last_retrieved = memory_last_retrieved.get(memory_id, "從未檢索")
                
                # 獲取用戶ID和創建時間
                user_id = doc.metadata.get("user_id", "unknown_user")
                created_time = doc.metadata.get("timestamp", time.time())
                created_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_time))
                
                # 創建符合新格式的記憶數據
                memory_data = {
                    "idx": idx,
                    "id": memory_id,
                    "user_id": user_id,
                    "text": doc.page_content,
                    "content": doc.page_content,  # 添加 content 欄位
                    "vector": [float(v) for v in vector],  # 確保向量中的所有值都是 float 類型
                    "created_time": created_time_str,
                    "last_retrieved": last_retrieved
                }
                
                # 檢查記憶是否已存在
                existing_memory = memory_collection.find_one({"id": memory_id})
                if existing_memory:
                    # 更新現有記憶
                    memory_collection.update_one(
                        {"id": memory_id},
                        {"$set": memory_data}
                    )
                    logging.info(f"更新記憶 ID: {memory_id}")
                else:
                    # 添加新記憶
                    new_memories.append(memory_data)
                    logging.info(f"添加新記憶 ID: {memory_id}")
            except Exception as e:
                logging.error(f"處理記憶 {getattr(doc, 'id', 'unknown')} 時出錯: {e}")
                continue
        
        # 批量插入新記憶
        if new_memories:
            memory_collection.insert_many(new_memories)
            logging.info(f"已將 {len(new_memories)} 條新記憶寫入 MongoDB")
    except Exception as e:
        logging.error(f"導出記憶到 MongoDB 時出錯: {e}")

# 從 MongoDB 加載聊天歷史
def load_chat_history_from_mongodb(user_id: str, thread_id: int = 1) -> List[Dict[str, Any]]:
    """從 MongoDB 加載聊天歷史"""
    try:
        # 獲取 MongoDB 客戶端
        mongo_client = get_mongo_client()
        if mongo_client is None:
            logging.error("無法獲取 MongoDB 客戶端")
            return []
            
        db = mongo_client[DB_NAME]
        chat_collection = db[CHAT_COLLECTION]
        
        # 查詢指定用戶和線程的聊天歷史
        chats = list(chat_collection.find({"user_id": user_id, "thread_id": thread_id}).sort("timestamp", 1))
        
        # 處理 MongoDB ObjectId
        for chat in chats:
            if "_id" in chat:
                chat["_id"] = str(chat["_id"])
                
        return chats
    except Exception as e:
        logging.error(f"從 MongoDB 加載聊天歷史時發生錯誤: {str(e)}")
        return []

# 從 MongoDB 加載檢索結果
def load_retrieval_results_from_mongodb(user_id: str) -> List[Dict[str, Any]]:
    """從 MongoDB 加載檢索結果"""
    try:
        # 獲取 MongoDB 客戶端
        mongo_client = get_mongo_client()
        if mongo_client is None:
            logging.error("無法獲取 MongoDB 客戶端")
            return []
            
        db = mongo_client[DB_NAME]
        retrieval_collection = db[RETRIEVAL_COLLECTION]
        
        # 查詢指定用戶的檢索結果
        results = list(retrieval_collection.find({"user_id": user_id}))
        
        # 處理 MongoDB ObjectId
        for result in results:
            if "_id" in result:
                result["_id"] = str(result["_id"])
                
        return results
    except Exception as e:
        logging.error(f"從 MongoDB 加載檢索結果時發生錯誤: {str(e)}")
        return []

# 將 Langchain 消息轉換為標準格式
def convert_from_langchain_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """將 Langchain 消息轉換為標準格式"""
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
        else:
            # 處理其他類型的消息
            result.append({"role": "system", "content": str(msg.content) if hasattr(msg, "content") else str(msg)})
    return result

# 將標準格式消息轉換為 Langchain 消息
def convert_to_langchain_messages(messages: List[Dict[str, Any]]) -> List[BaseMessage]:
    """
    將標準消息格式轉換為 Langchain 消息格式
    
    Args:
        messages: 標準格式的消息列表
        
    Returns:
        Langchain 格式的消息列表
    """
    langchain_messages = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        
        # 確保內容是字符串類型
        if not isinstance(content, str):
            content = str(content)
        
        # 根據角色創建不同類型的消息
        if role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content=content))
        elif role == "system":
            langchain_messages.append(SystemMessage(content=content))
        else:
            # 默認為人類消息
            langchain_messages.append(HumanMessage(content=content))
    
    return langchain_messages

# 初始化 MongoDB 集合
def initialize_mongodb_collections(force_reinit=False, init_chat_history=False, init_memory_stream=False):
    """初始化 MongoDB 集合
    
    Args:
        force_reinit (bool, optional): 如果為 True，則清空所有集合。預設為 False。
        init_chat_history (bool, optional): 如果為 True，則只清空聊天歷史集合。預設為 False。
        init_memory_stream (bool, optional): 如果為 True，則只清空記憶流集合。預設為 False。
    """
    try:
        # 獲取 MongoDB 客戶端
        mongo_client = get_mongo_client()
        if mongo_client is None:
            logging.error("無法獲取 MongoDB 客戶端")
            return False
            
        db = mongo_client[DB_NAME]
        
        # 確保必要的集合存在
        collections = db.list_collection_names()
        
        # 記錄初始化開始
        logging.info(f"開始初始化 MongoDB 集合，force_reinit={force_reinit}, init_chat_history={init_chat_history}, init_memory_stream={init_memory_stream}")
        
        # 創建必要的集合
        collections_created = []
        if CHAT_COLLECTION not in collections:
            db.create_collection(CHAT_COLLECTION)
            collections_created.append(CHAT_COLLECTION)
            
        if MEMORY_STREAM_COLLECTION not in collections:
            db.create_collection(MEMORY_STREAM_COLLECTION)
            collections_created.append(MEMORY_STREAM_COLLECTION)
            
        if RETRIEVAL_COLLECTION not in collections:
            db.create_collection(RETRIEVAL_COLLECTION)
            collections_created.append(RETRIEVAL_COLLECTION)
            
        # 如果有創建新集合，記錄信息
        if collections_created:
            logging.info(f"已創建以下集合: {', '.join(collections_created)}")
        
        # 根據參數選擇性地清空集合
        if force_reinit:
            # 清空所有集合
            chat_result = db[CHAT_COLLECTION].delete_many({})
            memory_result = db[MEMORY_STREAM_COLLECTION].delete_many({})
            retrieval_result = db[RETRIEVAL_COLLECTION].delete_many({})
            
            logging.info(f"已清空所有集合: {CHAT_COLLECTION}({chat_result.deleted_count}條記錄), "
                        f"{MEMORY_STREAM_COLLECTION}({memory_result.deleted_count}條記錄), "
                        f"{RETRIEVAL_COLLECTION}({retrieval_result.deleted_count}條記錄)")
        elif init_chat_history:
            # 只清空聊天歷史集合
            chat_result = db[CHAT_COLLECTION].delete_many({})
            logging.info(f"已清空聊天歷史集合: {CHAT_COLLECTION}({chat_result.deleted_count}條記錄)")
        elif init_memory_stream:
            # 只清空記憶流集合
            memory_result = db[MEMORY_STREAM_COLLECTION].delete_many({})
            logging.info(f"已清空記憶流集合: {MEMORY_STREAM_COLLECTION}({memory_result.deleted_count}條記錄)")
        else:
            logging.info("保留現有集合數據，跳過清空操作")
        
        logging.info("MongoDB 集合初始化完成")
        return True
    except Exception as e:
        logging.error(f"初始化 MongoDB 集合時發生錯誤: {str(e)}", exc_info=True)
        return False

# 主要回應生成函數
def response_generate(user_id: str, input_text: str, thread_id: int = 1) -> Dict[str, Any]:
    """
    生成對用戶輸入的回應
    
    Args:
        user_id: 用戶ID，用於識別當前說話的用戶
        input_text: 用戶輸入的文本
        thread_id: 對話線程ID，默認為1
        
    Returns:
        代理的回應字符串
    """
    global model, model_with_tools, current_model_name, tokenizer
    start_time = time.time()
    timing = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "processing_times": {}
    }
    
    # 輪換模型
    model_start_time = time.time()
    try:
        # 每次調用都重新獲取模型，確保模型輪換
        model, current_model_name = get_model()
        if not model:
            return "暫停回覆：目前沒有可用的模型，請稍後再試。"
            
        model_with_tools = model.bind_tools(tools)
        tokenizer = tiktoken.encoding_for_model(current_model_name)
        logging.info(f"本次對話使用模型: {current_model_name}")
        timing["processing_times"]["model_loading"] = round(time.time() - model_start_time, 3)
    except Exception as e:
        logging.error(f"輪換模型時發生錯誤: {str(e)}")
        return "暫停回覆：目前沒有可用的模型，請稍後再試。"
    
    # 創建記憶圖
    graph_start_time = time.time()
    graph = create_memory_graph()
    timing["processing_times"]["graph_creation"] = round(time.time() - graph_start_time, 3)
    
    # 加載聊天歷史
    history_start_time = time.time()
    chat_history = load_chat_history_from_mongodb(user_id, thread_id)
    timing["processing_times"]["chat_history_loading"] = round(time.time() - history_start_time, 3)
    
    # 轉換為 Langchain 消息格式
    conversion_start_time = time.time()
    messages = []
    for chat in chat_history:
        messages.append({"role": "user", "content": chat.get("user_input", "")})
        messages.append({"role": "assistant", "content": chat.get("agent_output", "")})
    
    # 添加當前用戶輸入
    messages.append({"role": "user", "content": input_text})
    
    # 轉換為 Langchain 消息
    langchain_messages = convert_to_langchain_messages(messages)
    timing["processing_times"]["message_conversion"] = round(time.time() - conversion_start_time, 3)
    
    # 設置配置
    config = {"configurable": {"user_id": user_id, "thread_id": thread_id}}
    
    try:
        # 運行圖
        api_call_start_time = time.time()
        result = graph.invoke(
            {
                "messages": langchain_messages,
                "recall_memories": [],
            },
            config=config,
        )
        api_call_end_time = time.time()
        timing["processing_times"]["api_response"] = round(api_call_end_time - api_call_start_time, 3)
        
        # 獲取代理回應
        agent_response = result["messages"][-1]
        
        # 轉換為標準格式
        response_msg = {
            "role": "assistant", 
            "content": agent_response.content,
            "thread_id": thread_id
        }
        
        # 記錄處理時間
        end_time = time.time()
        processing_time = end_time - start_time
        timing["processing_times"]["total"] = f"{processing_time:.2f}s"
        timing["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        # 添加時間信息
        response_msg["timing"] = timing
        
        # 獲取當前使用的模型
        response_msg["model_used"] = current_model_name
        
        # 獲取檢索結果
        memory_search_start_time = time.time()
        retrieval_results = load_retrieval_results_from_mongodb(user_id)
        timing["processing_times"]["memory_search"] = round(time.time() - memory_search_start_time, 3)
        response_msg["memory_references"] = retrieval_results
        
        # 保存聊天歷史
        save_start_time = time.time()
        save_chat_history_to_mongodb(user_id, [
            {"role": "user", "content": input_text, "thread_id": thread_id}, 
            response_msg
        ])
        timing["processing_times"]["saving_operations"] = round(time.time() - save_start_time, 3)
        
        # 打印時間統計
        logging.info("\n時間統計：")
        for operation, duration in timing["processing_times"].items():
            logging.info(f"{operation}: {duration}秒")
        logging.info(f"總處理時間: {round(processing_time, 3)}秒")
        
        # 返回包含所有信息的 JSON 格式響應
        return {
            "content": response_msg["content"],
            "model": current_model_name,
            "timing": timing,
            "memory_references": retrieval_results,
            "thread_id": thread_id
        }
    except Exception as e:
        logging.error(f"生成回應時發生錯誤: {str(e)}")
        error_message = "抱歉，處理您的請求時發生錯誤，請稍後再試。"
        
        error_response = {
            "content": error_message,
            "model": current_model_name,
            "timing": timing,
            "memory_references": [],
            "thread_id": thread_id,
            "error": str(e)
        }
        
        # 保存聊天歷史
        save_chat_history_to_mongodb(user_id, [
            {"role": "user", "content": input_text, "thread_id": thread_id}, 
            {"role": "assistant", "content": error_message, "thread_id": thread_id}
        ])
        
        return error_response

# 初始化 MongoDB 集合
initialize_mongodb_collections()

# 示例用法
if __name__ == "__main__":
    user_id = "test_user"
    response = response_generate(user_id, "你好，我是新用戶")
    logging.info(f"用戶: 你好，我是新用戶")
    logging.info(f"助手: {response}")
