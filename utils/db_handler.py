import pymysql
from pymysql.cursors import DictCursor
from utils.config_handler import mysql_config

class MySQLManager:
    def __init__(self):
        # 请确保这里的密码是你连接 dd1 时用的那个
        self.config = {
            "host": "localhost",
            "user": "root",
            "password": mysql_config["password"], 
            "database": "rag_agent_db",
            "charset": "utf8mb4",
            "cursorclass": DictCursor
        }

    def _get_conn(self):
        """建立数据库连接"""
        return pymysql.connect(**self.config)

    # --- 聊天记录相关操作 ---
    def save_message(self, session_id, role, content):
        """将聊天记录存入数据库"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                sql = "INSERT INTO chat_history (session_id, role, content) VALUES (%s, %s, %s)"
                cursor.execute(sql, (session_id, role, content))
            conn.commit()
        finally:
            conn.close()

    def load_messages(self, session_id, limit=20):
        """从数据库读取最近的聊天记录"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                # 按时间升序读取，以便大模型理解上下文顺序
                sql = "SELECT role, content FROM chat_history WHERE session_id = %s ORDER BY created_at ASC LIMIT %s"
                cursor.execute(sql, (session_id, limit))
                return cursor.fetchall()
        finally:
            conn.close()

    # --- 知识库文件相关操作 ---
    def is_file_exists(self, md5_hash):
        """检查文件是否已经入库过"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                sql = "SELECT id FROM knowledge_files WHERE md5_hash = %s"
                cursor.execute(sql, (md5_hash,))
                return cursor.fetchone() is not None
        finally:
            conn.close()

    def add_file_record(self, file_name, md5_hash):
        """记录新入库的文件"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                sql = "INSERT INTO knowledge_files (file_name, md5_hash) VALUES (%s, %s)"
                cursor.execute(sql, (file_name, md5_hash))
            conn.commit()
        finally:
            conn.close()

# 实例化对象，方便直接导入使用
db_manager = MySQLManager()