import sys
import os

# ==========================================
# 0. 核心路径修复 (必须放在最前面)
# ==========================================
current_path = os.path.abspath(__file__)
tools_dir = os.path.dirname(current_path)     # 当前在 agent/tools
agent_dir = os.path.dirname(tools_dir)        # 上一级是 agent
project_root = os.path.dirname(agent_dir)     # 再上一级是项目根目录 HeiMaAgent1.22

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 禁用所有代理环境变量，避免梯子干扰本地服务或 SSL 校验
os.environ['no_proxy'] = '*'
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

import numpy as np
from utils.logger_handle import logger
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from agent.rewrite.prompt.rewrite_prompt import get_rewrite_prompt



class QueryRewriter:
    """
    用于多轮对话的指代消解与上下文补全
    把形如 "它有什么缺点" 结合上下文重写为 "T10扫地机有什么缺点"
    """
    def __init__(self):
        self.prompt = get_rewrite_prompt
        self.rewrite_prompt = PromptTemplate.from_template(self.prompt())

    def rewrite(self, query: str, history: list) -> str:
        # 如果没有历史记录，直接返回原问题
        if not history:
            return query

        # 只需要提取最近的 3~5 轮对话作为参考即可，过滤掉 system 提示词
        recent_history = [m for m in history if m['role'] in ['user', 'assistant']][-6:]
        
        # 将 history 列表格式化为纯文本
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in recent_history])

        try:
            # 调用大模型进行极速重写 (非流式)
            result = chat_model.invoke(self.rewrite_prompt.format(history=history_text, query=query))
            rewritten_query = result.content.strip()
            
            # 清洗大模型可能输出的废话前缀
            if "：" in rewritten_query:
                rewritten_query = rewritten_query.split("：")[-1].strip()
            if ":" in rewritten_query:
                rewritten_query = rewritten_query.split(":")[-1].strip()
                
            logger.info(f"[Query重写] 原问题: 【{query}】 -> 重写后: 【{rewritten_query}】")
            return rewritten_query
            
        except Exception as e:
            logger.error(f"[Query重写] 重写失败，降级使用原问题: {e}")
            return query
        
