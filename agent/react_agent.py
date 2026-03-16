import sys
import os 

# --- 核心路径修复 ---

# --- 核心路径修复 ---
current_path = os.path.abspath(__file__)
rag_dir = os.path.dirname(current_path)
project_root = os.path.dirname(rag_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from model.factory import chat_model
from agent.router.router import HybridIntentRouter
from utils.history_manager import history_manager
from utils.prompt_loader import load_system_prompt

# 导入你原有的工具和中间件
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch

# 导入你之前写好的 RAG 服务
from rag.rag_summerize import RagSummarizeService

class SmartRoutingAgent:
    def __init__(self):
        # 1. 初始化原始的重量级 ReAct Agent (用于复杂任务)
        self.heavy_agent = create_agent(
            model= chat_model,
            system_prompt=load_system_prompt(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )
        self.router = HybridIntentRouter()
        
        # 2. 初始化 RAG 服务 (专职处理知识库问答)
        self.rag_service = RagSummarizeService()
        

    def execute_stream(self, query: str, session_id: str = "default"):
        """主入口：先路由，再分发执行"""
        
        # --- 第 1 步：意图识别 ---
        # 这一步是非流式的，但因为 Prompt 极短，通常在 0.5 秒内返回
        intent = self.router.predict_intent(query)
        print(f"\n[系统日志] 意图识别结果: => {intent} <=\n")

        # --- 第 2 步：加载历史记录 ---
        history_messages = history_manager.load_history(session_id) or []

        # --- 第 3 步：根据意图路由到不同的处理逻辑 ---
        full_response = ""
        
        if "chat" in intent:
            # 通道 A：闲聊模式，直接用大模型对话，速度极快
            input_messages = history_messages + [{"role": "user", "content": query}]
            for chunk in chat_model.stream(input_messages):
                if chunk.content:
                    full_response += chunk.content
                    yield chunk.content
                    
        elif "rag" in intent:
            # 通道 B：单纯的 RAG 问答，不走 Agent 思考链
            rag_result = self.rag_service.rag_summarize(query)
            for char in rag_result: # 简单的逐字模拟流式
                full_response += char
                yield char
                
        else:
            # 通道 C：默认走重量级 Agent (生成报告、查天气等)
            input_messages = history_messages + [{"role": "user", "content": query}]
            input_dict = {"messages": input_messages}
            
            for chunk in self.heavy_agent.stream(input_dict, stream_mode="messages", context={"report": False}):
                message_chunk = chunk[0] if isinstance(chunk, tuple) else chunk
                
                # 修改点 2：安全获取内容
                if hasattr(message_chunk, 'content'):
                    content = str(message_chunk.content)
                    if content:
                        full_response += content
                        yield content
                # latest_message = chunk["messages"][-1]
                # # 过滤出 AI 的回复进行流式输出
                # if isinstance(latest_message, AIMessage) and latest_message.content:
                #     chunk_content = latest_message.content.strip() + "\n"
                #     full_response += chunk_content
                #     yield chunk_content

        # --- 第 4 步：保存历史记录 (统一处理) ---
        updated_history = history_messages + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": full_response}
        ]
        
        # 优化点：保留最近的对话记录，但不暴力切片，确保偶数（一问一答完整性）
        max_history_length = 20 
        if len(updated_history) > max_history_length:
            updated_history = updated_history[-max_history_length:]
            # 确保截断后的第一条一定是 user
            if updated_history[0]["role"] != "user":
                updated_history = updated_history[1:]
                
        history_manager.save_history(session_id, updated_history)

if __name__ == '__main__':
    agent = SmartRoutingAgent()

    # print("\n--- 测试 1：闲聊 ---")
    # for chunk in agent.execute_stream("你好，你是谁？"):
    #     print(chunk, end="", flush=True)

    print("\n--- 测试 2：RAG 问答 ---")
    for chunk in agent.execute_stream("小户型适合哪种扫地机器人"):
        print(chunk, end="", flush=True)
    print("此处是分割部分，上面是 RAG 模式的输出，下面是复杂任务模式的输出")
    print("\n--- 测试 3：复杂任务 ---")
    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)