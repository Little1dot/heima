import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

# 导入你写好的 Agent 和 DB
from agent.react_agent import SmartRoutingAgent
from utils.db_handler import db_manager

# 1. 初始化 FastAPI 应用
app = FastAPI(title="智扫通 Agent API", description="企业级流式智能客服接口")

# 2. 配置跨域（允许前端 Vue/React 等跨域调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 初始化全局的 Agent 实例（常驻内存，避免每次请求重复初始化）
agent_instance = SmartRoutingAgent()

# 4. 定义前端传入的请求体格式 (Pydantic)
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"

# ==========================================
# 核心接口：SSE 流式对话接口
# ==========================================
@app.post("/v1/chat/completions")
async def chat_stream(request: ChatRequest):
    """
    处理用户聊天请求，并以 SSE (Server-Sent Events) 格式流式返回数据
    """
    query = request.query
    session_id = request.session_id

    # 定义生成器函数，桥接 Agent 的流式输出
    async def event_generator():
        try:
            # 迭代你的 agent.execute_stream (注意：目前你的 execute_stream 是同步的生成器)
            # 在真实的异步高并发场景下，可以用 asyncio.to_thread 包装，这里为了兼容你的代码直接调用
            for chunk in agent_instance.execute_stream(query, session_id):
                # 按照 SSE 标准格式返回数据
                yield {
                    "event": "message",
                    "data": chunk
                }
            # 结束标志
            yield {
                "event": "done",
                "data": "[DONE]"
            }
        except Exception as e:
            yield {
                "event": "error",
                "data": str(e)
            }

    # 返回 SSE 响应体
    return EventSourceResponse(event_generator())

# ==========================================
# 辅助接口：获取历史会话列表
# ==========================================
@app.get("/v1/sessions")
async def get_sessions():
    """获取数据库中所有的历史会话 ID"""
    sessions = db_manager.list_sessions()
    return {"code": 200, "data": sessions}

if __name__ == "__main__":
    # 使用 Uvicorn 启动 ASGI 服务器
    print("🚀 启动 FastAPI 服务端，监听端口 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)