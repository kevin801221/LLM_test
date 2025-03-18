// 全局變量
const USER_ID = 'default_user'; // 默認用戶ID
let hoverTimer = null; // 懸停計時器

// DOM 元素
const chatWindow = document.getElementById('chatWindow');
const memoryList = document.getElementById('memoryList');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const memoryPopup = document.getElementById('memoryPopup');
const memoryPopupContent = document.getElementById('memoryPopupContent');
const closePopup = document.getElementById('closePopup');

// 頁面加載完成後初始化
document.addEventListener('DOMContentLoaded', () => {
    // 載入聊天歷史
    loadChatHistory();
    
    // 載入記憶流
    loadMemoryStream();
    
    // 設置事件監聽器
    setupEventListeners();
});

// 設置事件監聽器
function setupEventListeners() {
    // 發送按鈕點擊事件
    sendButton.addEventListener('click', sendMessage);
    
    // 輸入框按下 Enter 鍵事件
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // 關閉記憶參考浮動視窗
    closePopup.addEventListener('click', hidePopup);
    
    // 點擊其他地方關閉浮動視窗
    document.addEventListener('click', (e) => {
        if (!memoryPopup.contains(e.target) && 
            !e.target.classList.contains('agent-message-content')) {
            hidePopup();
        }
    });
}

// 隱藏浮動視窗
function hidePopup() {
    memoryPopup.style.display = 'none';
}

// 載入聊天歷史
async function loadChatHistory() {
    try {
        const response = await fetch(`/api/chat_history?user_id=${USER_ID}`);
        const data = await response.json();
        
        if (data.chat_history && data.chat_history.length > 0) {
            // 清空聊天窗口
            chatWindow.innerHTML = '';
            
            // 添加對話到聊天窗口
            data.chat_history.forEach(chat => {
                addMessageToChat('user', chat.user_input);
                addMessageToChat('agent', chat.agent_output, chat.memory_references, chat.model_used, chat.timing);
            });
            
            // 滾動到底部
            scrollToBottom();
        }
    } catch (error) {
        console.error('載入聊天歷史時發生錯誤:', error);
    }
}

// 載入記憶流
async function loadMemoryStream() {
    try {
        const response = await fetch(`/api/memories?user_id=${USER_ID}`);
        const data = await response.json();
        
        if (data.memories && data.memories.length > 0) {
            // 清空記憶列表
            memoryList.innerHTML = '';
            
            // 添加記憶到記憶列表
            data.memories.forEach(memory => {
                addMemoryToList(memory);
            });
        }
    } catch (error) {
        console.error('載入記憶流時發生錯誤:', error);
    }
}

// 發送訊息
async function sendMessage() {
    const message = userInput.value.trim();
    
    if (message) {
        // 添加用戶訊息到聊天窗口
        addMessageToChat('user', message);
        
        // 清空輸入框
        userInput.value = '';
        
        // 顯示載入中提示
        const loadingId = addLoadingMessage();
        
        try {
            // 發送請求到伺服器
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    user_id: USER_ID
                }),
            });
            
            const data = await response.json();
            
            // 移除載入中提示
            removeLoadingMessage(loadingId);
            
            // 添加代理回應到聊天窗口
            addMessageToChat('agent', data.response, data.memory_references, data.model_used, data.timing);
            
            // 重新載入記憶流
            loadMemoryStream();
            
            // 滾動到底部
            scrollToBottom();
        } catch (error) {
            console.error('發送訊息時發生錯誤:', error);
            
            // 移除載入中提示
            removeLoadingMessage(loadingId);
            
            // 添加錯誤訊息
            addMessageToChat('agent', '抱歉，處理您的請求時發生錯誤，請稍後再試。');
            
            // 滾動到底部
            scrollToBottom();
        }
    }
}

// 添加訊息到聊天窗口
function addMessageToChat(role, content, memoryRefs = [], modelUsed = '', timing = null) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', `${role}-message`);
    
    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content', `${role}-message-content`);
    contentDiv.textContent = content;
    
    // 如果是代理訊息，添加記憶參考點擊事件
    if (role === 'agent') {
        contentDiv.dataset.memoryRefs = JSON.stringify(memoryRefs);
        contentDiv.dataset.modelUsed = modelUsed;
        contentDiv.dataset.timing = JSON.stringify(timing);
        
        // 添加滑鼠事件
        contentDiv.addEventListener('mouseenter', function() {
            // 清除之前的計時器
            if (hoverTimer) clearTimeout(hoverTimer);
            
            // 設置新的計時器，1秒後顯示浮動視窗
            hoverTimer = setTimeout(() => {
                showMemoryReferences(this);
            }, 1000);
        });
        
        contentDiv.addEventListener('mouseleave', function() {
            // 清除計時器
            if (hoverTimer) clearTimeout(hoverTimer);
        });
        
        // 添加浮動視窗的滑鼠事件
        memoryPopup.addEventListener('mouseenter', function() {
            // 當滑鼠進入浮動視窗時，不要隱藏
            if (hoverTimer) clearTimeout(hoverTimer);
        });
        
        memoryPopup.addEventListener('mouseleave', function() {
            // 當滑鼠離開浮動視窗時，隱藏
            hidePopup();
        });
    }
    
    messageDiv.appendChild(contentDiv);
    chatWindow.appendChild(messageDiv);
}

// 添加記憶到記憶列表
function addMemoryToList(memory) {
    const memoryDiv = document.createElement('div');
    memoryDiv.classList.add('memory-item');
    memoryDiv.dataset.id = memory.id;
    
    const textDiv = document.createElement('div');
    textDiv.classList.add('memory-text');
    textDiv.textContent = memory.text;
    
    const metaDiv = document.createElement('div');
    metaDiv.classList.add('memory-meta');
    
    const timeSpan = document.createElement('span');
    const createdDate = new Date(memory.created_time);
    timeSpan.textContent = `創建: ${createdDate.toLocaleString()}`;
    
    const retrievedSpan = document.createElement('span');
    const lastRetrievedDate = new Date(memory.last_retrieved);
    retrievedSpan.textContent = `上次檢索: ${lastRetrievedDate.toLocaleString()}`;
    
    metaDiv.appendChild(timeSpan);
    metaDiv.appendChild(retrievedSpan);
    
    memoryDiv.appendChild(textDiv);
    memoryDiv.appendChild(metaDiv);
    
    memoryList.appendChild(memoryDiv);
}

// 添加載入中提示
function addLoadingMessage() {
    const id = 'loading-' + Date.now();
    const loadingDiv = document.createElement('div');
    loadingDiv.id = id;
    loadingDiv.classList.add('message', 'agent-message', 'loading-message');
    
    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content', 'agent-message-content');
    
    const loadingText = document.createElement('span');
    loadingText.textContent = '正在思考';
    
    const dotsContainer = document.createElement('span');
    dotsContainer.classList.add('loading-dots');
    dotsContainer.textContent = '...';
    
    contentDiv.appendChild(loadingText);
    contentDiv.appendChild(dotsContainer);
    loadingDiv.appendChild(contentDiv);
    
    chatWindow.appendChild(loadingDiv);
    scrollToBottom();
    
    return id;
}

// 移除載入中提示
function removeLoadingMessage(id) {
    const loadingElement = document.getElementById(id);
    if (loadingElement) {
        loadingElement.remove();
    }
}

// 顯示記憶參考浮動視窗
function showMemoryReferences(element) {
    const memoryRefsStr = element.dataset.memoryRefs;
    const modelUsed = element.dataset.modelUsed;
    const timingStr = element.dataset.timing;
    
    // 清空浮動視窗內容
    memoryPopupContent.innerHTML = '';
    
    // 添加模型信息
    if (modelUsed) {
        const modelDiv = document.createElement('div');
        modelDiv.classList.add('model-info');
        modelDiv.textContent = `使用模型: ${modelUsed}`;
        memoryPopupContent.appendChild(modelDiv);
    }
    
    // 添加處理時間信息
    if (timingStr) {
        try {
            const timing = JSON.parse(timingStr);
            const timingDiv = document.createElement('div');
            timingDiv.classList.add('timing-info');
            
            // 創建表格來顯示時間信息
            const table = document.createElement('table');
            table.classList.add('timing-table');
            
            // 添加表頭
            const thead = document.createElement('thead');
            const headerRow = document.createElement('tr');
            const header1 = document.createElement('th');
            header1.textContent = '處理階段';
            const header2 = document.createElement('th');
            header2.textContent = '時間 (秒)';
            headerRow.appendChild(header1);
            headerRow.appendChild(header2);
            thead.appendChild(headerRow);
            table.appendChild(thead);
            
            // 添加表體
            const tbody = document.createElement('tbody');
            
            // 處理所有時間信息
            if (timing.processing_times) {
                const processingTimes = timing.processing_times;
                const timeEntries = [
                    ['圖創建', 'graph_creation'],
                    ['聊天歷史加載', 'chat_history_loading'],
                    ['消息轉換', 'message_conversion'],
                    ['API響應', 'api_response'],
                    ['記憶搜索', 'memory_search'],
                    ['保存操作', 'saving_operations'],
                    ['總處理時間', 'total']
                ];

                timeEntries.forEach(([label, key]) => {
                    if (processingTimes[key] !== undefined) {
                        const row = document.createElement('tr');
                        const cell1 = document.createElement('td');
                        cell1.textContent = label;
                        const cell2 = document.createElement('td');
                        cell2.textContent = typeof processingTimes[key] === 'string' 
                            ? processingTimes[key] 
                            : processingTimes[key].toFixed(3);
                        row.appendChild(cell1);
                        row.appendChild(cell2);
                        tbody.appendChild(row);
                    }
                });
            }
            
            table.appendChild(tbody);
            timingDiv.appendChild(table);
            memoryPopupContent.appendChild(timingDiv);
        } catch (error) {
            console.error('解析處理時間信息時發生錯誤:', error);
        }
    }
    
    // 添加記憶參考
    if (memoryRefsStr) {
        try {
            const memoryRefs = JSON.parse(memoryRefsStr);
            
            if (memoryRefs.length > 0) {
                // 添加分隔線
                const divider = document.createElement('div');
                divider.classList.add('popup-divider');
                memoryPopupContent.appendChild(divider);
                
                // 添加記憶參考標題
                const titleDiv = document.createElement('div');
                titleDiv.classList.add('memory-references-title');
                titleDiv.textContent = '記憶參考:';
                memoryPopupContent.appendChild(titleDiv);
                
                // 添加記憶參考到浮動視窗
                memoryRefs.forEach(ref => {
                    const refDiv = document.createElement('div');
                    refDiv.classList.add('memory-reference');
                    
                    const contentDiv = document.createElement('div');
                    contentDiv.classList.add('memory-reference-content');
                    contentDiv.textContent = ref.content;
                    
                    const scoreDiv = document.createElement('div');
                    scoreDiv.classList.add('memory-reference-score');
                    scoreDiv.textContent = `相似度分數: ${ref.score.toFixed(4)}`;
                    
                    refDiv.appendChild(contentDiv);
                    refDiv.appendChild(scoreDiv);
                    
                    memoryPopupContent.appendChild(refDiv);
                });
            }
        } catch (error) {
            console.error('解析記憶參考時發生錯誤:', error);
        }
    }
    
    // 計算浮動視窗位置
    const rect = element.getBoundingClientRect();
    
    // 先設置為隱藏但可以計算尺寸
    memoryPopup.style.display = 'block';
    memoryPopup.style.visibility = 'hidden';
    
    // 獲取視窗和浮動視窗的尺寸
    const viewportHeight = window.innerHeight;
    const popupHeight = memoryPopup.offsetHeight;
    
    // 決定浮動視窗應該顯示在元素的上方還是下方
    const spaceBelow = viewportHeight - rect.bottom;
    const spaceAbove = rect.top;
    
    // 如果下方空間不足但上方空間足夠，則顯示在上方
    if (spaceBelow < popupHeight && spaceAbove > popupHeight) {
        memoryPopup.style.top = `${rect.top + window.scrollY - popupHeight - 10}px`;
    } else {
        memoryPopup.style.top = `${rect.bottom + window.scrollY + 10}px`;
    }
    
    // 確保浮動視窗不會超出視窗左右邊界
    const viewportWidth = window.innerWidth;
    const popupWidth = memoryPopup.offsetWidth;
    
    let leftPosition = rect.left + window.scrollX;
    if (leftPosition + popupWidth > viewportWidth) {
        leftPosition = viewportWidth - popupWidth - 10;
    }
    if (leftPosition < 10) {
        leftPosition = 10;
    }
    
    memoryPopup.style.left = `${leftPosition}px`;
    
    // 顯示浮動視窗
    memoryPopup.style.visibility = 'visible';
}

// 滾動到底部
function scrollToBottom() {
    chatWindow.scrollTop = chatWindow.scrollHeight;
}
