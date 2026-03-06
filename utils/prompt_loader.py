from utils.config_handler import prompts_config
from utils.path_tool import get_abs_path
from utils.logger_handle import logger

def load_system_prompt():
    try: 
        system_prompt_path = get_abs_path(prompts_config["main_prompt_path"])
    except KeyError as e: 
        logger.error(f"[load_system_prompt]未找到系统提示文件[main_prompt_path]")
        raise e 
    # 读取文件
    try:
        return open(system_prompt_path, "r", encoding="utf-8").read()
    except Exception as e: 
        logger.error(f"[load_system_prompt]解析系统提示词出错。{str(e)}")
        raise e 
    

def load_rag_prompt():
    try: 
        rag_prompt_path = get_abs_path(prompts_config["rag_summarize_prompt_path"])
    except KeyError as e: 
        logger.error(f"[load_rag_prompt]未找到rag提示文件[rag_summarize_prompt_path]")
        raise e 
    # 读取文件
    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except Exception as e: 
        logger.error(f"[load_rag_prompt]解析rag提示词出错。{str(e)}")
        raise e 
    
def load_report_prompt():
    # 获取报告提示文件路径
    try: 
        report_prompt_path = get_abs_path(prompts_config["report_prompt_path"])
    except KeyError as e: 
        logger.error(f"[load_report_prompt]未找到报告提示文件[report_prompt_path]")
        raise e 
    # 读取文件
    try:
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except Exception as e: 
        logger.error(f"[load_report_prompt]解析报告提示词出错。{str(e)}")
        raise e 

def load_classify_prompt():
    # 获取识别提示文件路径
    try: 
        classify_prompt_path = get_abs_path(prompts_config["classify_prompt_path"])
    except KeyError as e: 
        logger.error(f"[load_classify_prompt]未找到识别提示文件[classify_prompt_path]")
        raise e 
    # 读取文件
    try:
        return open(classify_prompt_path, "r", encoding="utf-8").read()
    except Exception as e: 
        logger.error(f"[load_classify_prompt]解析识别提示词出错。{str(e)}")
        raise e 

def load_history_message_summarize_prompt():
    # 获取历史消息总结提示文件路径
    try: 
        history_summarize_prompt_path = get_abs_path(prompts_config["history_summarize_prompt_path"])
    except KeyError as e: 
        logger.error(f"[load_history_message_summarize_prompt]未找到历史消息总结提示文件[history_summarize_prompt_path]")
        raise e 
    # 读取文件
    try:
        return open(history_summarize_prompt_path, "r", encoding="utf-8").read()
    except Exception as e: 
        logger.error(f"[load_history_message_summarize_prompt]解析历史消息总结提示词出错。{str(e)}")
        raise e
    