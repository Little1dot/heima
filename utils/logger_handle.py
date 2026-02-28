import logging
from utils.path_tool import get_abs_path
import os

# 日志保存的根目录
LOG_ROOT = get_abs_path("log")

# 确保日志的目录存在
os.makedirs(LOG_ROOT, exist_ok=True)

# 日志的格式配置
# %(asctime)s时间戳 %(name)s记录器名 %(levelname)s日志级别 %(filename)s文件名 %(lineno)d行号 %(message)s消息内容
DEFULT_LOG_FORMAT = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

def get_logger(name: str= "agent",
               console_level: int = logging.INFO,
               file_level:int = logging.DEBUG,
               log_file = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加handler，多文件使用，避免重复导入日志
    if logger.handlers:
        return logger
    
    # 配置handler器
    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFULT_LOG_FORMAT)

    logger.addHandler(console_handler)
    
    # 文件输出
    if not log_file:
        log_file = os.path.join(LOG_ROOT, f"{name}.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFULT_LOG_FORMAT)

    logger.addHandler(file_handler)

    return logger

# 便捷获取日志对象
logger = get_logger()

if __name__ == "__main__":
    logger.info("信息日志")
    logger.debug("调试日志")
    logger.warning("警告日志")
    logger.error("错误日志")