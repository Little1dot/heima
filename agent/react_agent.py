import sys
import os 
# --- 核心修复代码开始 ---
# 1. 获取当前文件(vector_store.py)的绝对路径
current_path = os.path.abspath(__file__)
# 2. 获取父目录(rag)
rag_dir = os.path.dirname(current_path)
# 3. 获取爷目录(HeiMaAgent1.22)，也就是项目的根目录
project_root = os.path.dirname(rag_dir)

# 4. 把根目录加入到 Python 的搜索路径中 (插在第一个位置)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompt
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from utils.history_manager import history_manager


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompt(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    # 流式执行
    def execute_stream(self, query: str, session_id: str = "default"):
        # 加载历史消息
        history_messages = history_manager.load_history(session_id)
        
        # 构建包含历史消息的输入
        input_messages = history_messages + [
            {"role": "user", "content": query},
        ]
        
        input_dict = {
            "messages": input_messages
        }

        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        full_response = ""
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                chunk_content = latest_message.content.strip() + "\n"
                full_response += chunk_content
                yield chunk_content
        
        # 保存更新后的历史消息
        updated_history = history_messages + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": full_response}
        ]
        # 限制历史消息数量，避免过长
        max_history_length = 20  # 保留最近10轮对话（20条消息）
        if len(updated_history) > max_history_length:
            updated_history = updated_history[-max_history_length:]
        history_manager.save_history(session_id, updated_history)


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
