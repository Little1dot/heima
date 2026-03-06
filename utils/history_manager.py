import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from utils.path_tool import get_abs_path
from utils.logger_handle import logger
from utils.prompt_loader import load_history_message_summarize_prompt

# 【新增导包】：引入大模型和提示词模板，用于生成历史摘要
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model

class HistoryManager:
    """
    历史消息管理器，用于存储、加载对话历史，并支持大模型动态记忆压缩
    """
    def __init__(self, history_dir: str = "data/history", max_tokens: int = 1500):
        # 获取历史消息存储目录的绝对路径
        self.history_dir = get_abs_path(history_dir)
        # 确保目录存在
        os.makedirs(self.history_dir, exist_ok=True)
        
        # 【新增】：设定安全上下文阈值，超过这个字数就触发压缩 (可根据需要调整)
        self.max_tokens = max_tokens
        self.load_history_message_summarize_prompt = load_history_message_summarize_prompt
        # 【新增】：定义摘要提取的 Prompt
        self.summary_prompt = PromptTemplate.from_template(
            self.load_history_message_summarize_prompt()
        )
    
    # ---------------- 原始的读写存储逻辑 (保持不变) ----------------
    def get_history_file_path(self, session_id: str) -> str:
        return os.path.join(self.history_dir, f"{session_id}.json")
    
    def save_history(self, session_id: str, messages: List[Dict]) -> bool:
        try:
            file_path = self.get_history_file_path(session_id)
            history_data = {
                "timestamp": datetime.now().isoformat(),
                "messages": messages
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
            logger.info(f"[历史消息]会话{session_id}的历史消息已保存")
            return True
        except Exception as e:
            logger.error(f"[历史消息]保存历史消息失败: {e}")
            return False
    
    def load_history(self, session_id: str) -> Optional[List[Dict]]:
        try:
            file_path = self.get_history_file_path(session_id)
            if not os.path.exists(file_path):
                return []
            with open(file_path, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            return history_data.get("messages", [])
        except Exception as e:
            logger.error(f"[历史消息]加载历史消息失败: {e}")
            return []
            
    def delete_history(self, session_id: str) -> bool:
        try:
            file_path = self.get_history_file_path(session_id)
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"[历史消息]会话{session_id}的历史消息已删除")
            return True
        except Exception as e:
            logger.error(f"[历史消息]删除历史消息失败: {e}")
            return False

    def list_sessions(self) -> List[str]:
        try:
            sessions = []
            for file_name in os.listdir(self.history_dir):
                if file_name.endswith('.json'):
                    session_id = file_name[:-5]
                    sessions.append(session_id)
            return sessions
        except Exception as e:
            logger.error(f"[历史消息]列出会话失败: {e}")
            return []

    # ---------------- 【核心新增】大模型记忆压缩优化逻辑 ----------------
    def _estimate_tokens(self, text: str) -> int:
        """粗略估算 Token 长度 (生产环境常用 tiktoken，这里用中文字符长度平替)"""
        return len(text)

    def get_optimized_history(self, session_id: str) -> List[Dict]:
        """
        获取经过动态压缩优化后的历史上下文，防止 Token 溢出
        """
        # 1. 从 JSON 文件加载全部的原始历史记录
        raw_messages = self.load_history(session_id)
        if not raw_messages:
            return []

        recent_msgs = []
        old_msgs_for_summary = []
        current_tokens = 0

        # 2. 从最新的消息开始逆序遍历，计算累加 Token
        for msg in reversed(raw_messages):
            msg_len = self._estimate_tokens(msg["content"])
            # 如果还没到达阈值，放入“近期消息”池，保持原汁原味
            if current_tokens + msg_len <= self.max_tokens:
                recent_msgs.insert(0, msg)
                current_tokens += msg_len
            else:
                # 超出阈值的老消息，放入“待摘要”池
                old_msgs_for_summary.insert(0, msg)

        final_context = []
        
        # 3. 如果存在古老消息，触发大模型进行记忆压缩
        if old_msgs_for_summary:
            logger.info(f"[记忆管理] 会话 {session_id} 历史超出 {self.max_tokens} 字，触发大模型摘要压缩机制...")
            # 拼接老对话
            history_text = "\n".join([f"{m['role']}: {m['content']}" for m in old_msgs_for_summary])
            
            try:
                # 调用大模型生成摘要 (非流式调用即可)
                summary_result = chat_model.invoke(self.summary_prompt.format(history=history_text))
                summary = summary_result.content
                logger.info(f"[记忆管理] ✨ 压缩完成，提取核心记忆: {summary}")
                
                # 将压缩后的摘要伪装成 System 提示词，注入到最终上下文的最顶端
                final_context.append({"role": "system", "content": f"以下是用户之前的历史记忆摘要，请作为背景参考：\n{summary}"})
            except Exception as e:
                logger.error(f"[记忆管理] 摘要生成失败，跳过老旧记忆: {e}")

        # 4. 拼接最近的原始对话并返回
        final_context.extend(recent_msgs)
        return final_context

# 创建全局历史管理器实例
history_manager = HistoryManager(max_tokens=1500)