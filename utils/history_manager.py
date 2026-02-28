import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from utils.path_tool import get_abs_path
from utils.logger_handle import logger

class HistoryManager:
    """
    历史消息管理器，用于存储和加载对话历史
    """
    def __init__(self, history_dir: str = "data/history"):
        # 获取历史消息存储目录的绝对路径
        self.history_dir = get_abs_path(history_dir)
        # 确保目录存在
        os.makedirs(self.history_dir, exist_ok=True)
    
    def get_history_file_path(self, session_id: str) -> str:
        """
        获取历史消息文件路径
        """
        return os.path.join(self.history_dir, f"{session_id}.json")
    
    def save_history(self, session_id: str, messages: List[Dict]) -> bool:
        """
        保存历史消息到文件
        """
        try:
            file_path = self.get_history_file_path(session_id)
            # 添加时间戳
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
        """
        从文件加载历史消息
        """
        try:
            file_path = self.get_history_file_path(session_id)
            if not os.path.exists(file_path):
                logger.warning(f"[历史消息]会话{session_id}的历史消息文件不存在")
                return []
            with open(file_path, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            logger.info(f"[历史消息]会话{session_id}的历史消息已加载")
            return history_data.get("messages", [])
        except Exception as e:
            logger.error(f"[历史消息]加载历史消息失败: {e}")
            return []
    
    def delete_history(self, session_id: str) -> bool:
        """
        删除历史消息文件
        """
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
        """
        列出所有会话ID
        """
        try:
            sessions = []
            for file_name in os.listdir(self.history_dir):
                if file_name.endswith('.json'):
                    session_id = file_name[:-5]  # 移除.json后缀
                    sessions.append(session_id)
            return sessions
        except Exception as e:
            logger.error(f"[历史消息]列出会话失败: {e}")
            return []

# 创建全局历史管理器实例
history_manager = HistoryManager()