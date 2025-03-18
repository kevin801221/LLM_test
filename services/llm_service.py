#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
from datetime import datetime
from openai import OpenAI
import ollama
from utils.api_cost_tracker import track_openai_cost
from utils.conversation.mongodb_memory_agent import response_generate

class LLMService:
    """Service for handling LLM interactions (OpenAI or Ollama)"""
    
    def __init__(self, model_name="gpt4o"): 
        """Initialize LLM service with specified model"""
        self.model_name = model_name
        
        # Determine which LLM to use based on model name
        if model_name in ["gpt4o", "gpt-4o", "gpt-4"]:
            self.use_openai = True
            self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            print(f"使用 OpenAI {model_name} 模型")
        else:
            self.use_openai = False
            # Ollama model will be used directly with model_name
            print(f"使用 Ollama {model_name} 模型")
    
    def generate_prompt(self, employee_data, recent_messages=None, is_first_chat=True):
        """Generate prompt based on employee data and conversation history"""
        try:
            if is_first_chat:
                prompt = f"""你現在是 YCM 館長，一個友善、專業的智能助手。你正在與 {employee_data.get('name', '訪客')} 進行對話。

你應該：
1. 以簡短的問候開始對話
2. 保持專業但友善的態度
3. 給出簡短的回應，不要太長
"""
            else:
                prompt = f"""你現在是 YCM 館長，一個友善、專業的智能助手。你正在與 {employee_data.get('name', '訪客')} 進行對話。

你應該：
1. 根據用戶的輸入給出合適的回應
2. 保持專業但友善的態度
3. 給出簡短的回應，不要太長
"""

            # Add conversation history if available
            if recent_messages:
                prompt += "\n最近的對話記錄：\n"
                for role, message in recent_messages:
                    prompt += f"{role}: {message}\n"

            # Add detailed employee information if available
            if all(key in employee_data for key in ['name', 'chinese_name', 'department', 'position']):
                prompt += f"""
以下是關於 {employee_data['name']} 的資訊：

基本資料：
- 中文名字：{employee_data.get('chinese_name', '未提供')}
- 部門：{employee_data.get('department', '未提供')}
- 職位：{employee_data.get('position', '未提供')}
- 工作年資：{employee_data.get('total_years_experience', '未提供')} 年
"""

                if 'technical_skills' in employee_data and employee_data['technical_skills']:
                    prompt += f"\n專業技能：\n{', '.join(employee_data['technical_skills'])}"

                if 'interests' in employee_data and employee_data['interests']:
                    prompt += f"\n\n興趣愛好：\n{', '.join(employee_data['interests'])}"

                if 'certificates' in employee_data and employee_data['certificates']:
                    prompt += "\n\n證書：\n"
                    prompt += "\n".join([f"- {cert['name']} (由 {cert['issuing_organization']} 頒發)" 
                                    for cert in employee_data['certificates']])

                if 'work_experiences' in employee_data and employee_data['work_experiences']:
                    prompt += "\n\n工作經驗：\n"
                    prompt += "\n".join([f"- {exp['company_name']}: {exp['position']} ({exp['description']})" 
                                    for exp in employee_data['work_experiences']])

            print(f"生成的提示詞長度: {len(prompt)}")
            return prompt
        except Exception as e:
            print(f"生成提示詞時發生錯誤: {e}")
            return f"你是 YCM 館長，請友善地與用戶對話。"
    
    def chat_with_employee(self, employee_data, is_first_chat=True):
        """Start or continue conversation with an employee"""
        try:
            print(f"開始與員工對話，使用模型: {self.model_name}")
            
            # Generate initial prompt
            system_prompt = self.generate_prompt(employee_data, is_first_chat=is_first_chat)
            
            start_time = time.time()
            if self.use_openai:
                try:
                    # Use OpenAI API
                    response = self.openai_client.chat.completions.create(
                        model=self.model_name if self.model_name != "gpt4o" else "gpt-4o",
                        messages=[
                            {"role": "system", "content": system_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=300
                    )
                    result = response.choices[0].message.content
                    print(f"OpenAI 回應 ({time.time() - start_time:.2f}秒): {result}")
                    return result
                except Exception as e:
                    print(f"OpenAI API 錯誤: {e}")
                    return "抱歉，AI 服務暫時無法使用。"
            else:
                try:
                    # Use Ollama
                    response = ollama.chat(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt}
                        ]
                    )
                    result = response['message']['content']
                    print(f"Ollama 回應 ({time.time() - start_time:.2f}秒): {result}")
                    return result
                except Exception as e:
                    print(f"Ollama 錯誤: {e}")
                    return "抱歉，AI 服務暫時無法使用。"
        except Exception as e:
            print(f"對話系統錯誤: {e}")
            return "系統錯誤，請稍後再試。"
    
    def handle_user_message(self, employee_data, user_message, conversation_memory):
        """Process user message and generate response"""
        try:
            # Calculate message importance
            importance = conversation_memory.calculate_message_importance(user_message)
            
            # Record user message
            conversation_memory.add_message(
                employee_data['name'], 
                user_message, 
                'user',
                importance
            )
            
            # Get recent conversation history
            recent_messages = conversation_memory.get_recent_messages(employee_data['name'])
            
            # Generate system prompt
            system_prompt = self.generate_prompt(employee_data, recent_messages, is_first_chat=False)
            
            # Generate response
            start_time = time.time()
            if self.use_openai:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
                
                # 使用裝飾器追蹤 API 成本
                response = self._openai_chat_completion(
                    model=self.model_name if self.model_name != "gpt4o" else "gpt-4o",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=300
                )
                ai_response = response.choices[0].message.content
            else:
                messages = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_message}
                ]
                response = ollama.chat(
                    model=self.model_name,
                    messages=messages
                )
                ai_response = response['message']['content']
            
            print(f"AI 回應耗時: {time.time() - start_time:.2f}秒")
            
            # Record AI response
            conversation_memory.add_message(employee_data['name'], ai_response, 'assistant')
            
            return ai_response
            
        except Exception as e:
            print(f"處理用戶訊息時發生錯誤: {e}")
            return "抱歉，我現在無法正常回應。請稍後再試。"
    
    # 添加裝飾器來追蹤 OpenAI API 成本
    @track_openai_cost(api_type='chat')
    def _openai_chat_completion(self, **kwargs):
        """使用 OpenAI API 進行聊天完成，並追蹤成本"""
        return self.openai_client.chat.completions.create(**kwargs)
    
    def handle_user_message_with_search(self, employee_data, user_message, conversation_memory, search_results=None):
        """處理用戶消息並生成回應，可選擇性地包含搜索結果
        
        Args:
            employee_data: 員工數據
            user_message: 用戶消息
            conversation_memory: 對話記憶
            search_results: 搜索結果（可選）
            
        Returns:
            str: AI 回應
        """
        try:
            # 計算消息重要性
            importance = conversation_memory.calculate_message_importance(user_message)
            
            # 記錄用戶消息
            conversation_memory.add_message(
                employee_data['name'], 
                user_message, 
                'user',
                importance
            )
            
            # 使用 response_generate 生成回應
            ai_response = response_generate(employee_data['name'], user_message)
            
            # 記錄 AI 回應
            conversation_memory.add_message(employee_data['name'], ai_response, 'assistant')
            
            return ai_response
            
        except Exception as e:
            print(f"處理用戶訊息時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            return "抱歉，我現在無法正常回應。請稍後再試。"