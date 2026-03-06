import time
import uuid
import json
import requests
import streamlit as st

# ==========================================
# 1. 后端 API 配置
# ==========================================
API_BASE_URL = "http://localhost:8000/v1"

def get_sessions_from_api():
    """从后端 API 获取会话列表"""
    try:
        res = requests.get(f"{API_BASE_URL}/sessions", timeout=5)
        if res.status_code == 200:
            return res.json().get("data", [])
    except Exception as e:
        st.sidebar.error(f"无法连接后端服务: {e}")
    return []

def chat_stream_from_api(query: str, session_id: str):
    """请求后端 SSE 流式接口，并生成打字机效果"""
    url = f"{API_BASE_URL}/chat/completions"
    payload = {"query": query, "session_id": session_id}
    
    try:
        # 开启 HTTP 流式请求 (stream=True)
        with requests.post(url, json=payload, stream=True, timeout=60) as response:
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    # 解析 SSE 协议中的 data 字段
                    if decoded_line.startswith("data: "):
                        chunk = decoded_line[6:]  # 截取 "data: " 后面的内容
                        if chunk == "[DONE]":
                            break
                        # 模拟网络传输的极小延迟，让打字机效果更丝滑
                        time.sleep(0.01)
                        yield chunk
    except Exception as e:
        yield f"\n[请求后端异常]: {e}"

# ==========================================
# 2. 前端 UI 与交互逻辑
# ==========================================
st.set_page_config(page_title="智扫通智能客服", page_icon="🤖")
st.title("智扫通机器人智能客服")
st.caption("基于 FastAPI + SSE 流式微服务架构")
st.divider()

# 会话管理侧边栏
with st.sidebar:
    st.subheader("会话管理")
    
    # 动态从后端拉取历史会话
    sessions = get_sessions_from_api()
    
    selected_session = st.selectbox(
        "选择会话",
        options=["新建会话"] + sessions,
        key="session_selector"
    )
    
    # 状态初始化
    if selected_session == "新建会话":
        new_session_id = str(uuid.uuid4())[:8]
        st.session_state["session_id"] = new_session_id
        st.session_state["message"] = []
        st.info(f"已创建新会话: {new_session_id}")
    else:
        # 注意：这里为了极致的瘦客户端，实际上可以再写一个 API 专门拉取某个 session 的历史记录。
        # 这里为了演示流畅度，我们切换会话时暂时只保留本地 state，后端底层依旧是有记忆的。
        if "session_id" not in st.session_state or st.session_state["session_id"] != selected_session:
            st.session_state["session_id"] = selected_session
            st.session_state["message"] = [] # 清空前端展示，但后端 MySQL 里依然存有上下文
            st.info(f"已切换到会话: {selected_session}")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = "default"

if "message" not in st.session_state:
    st.session_state["message"] = []

# 渲染聊天记录
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入区域
prompt = st.chat_input("请描述您遇到的扫地机问题，或输入指令...")

if prompt:
    # 1. 前端展示用户消息
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    # 2. 请求后端 API 并流式渲染
    response_messages = []
    with st.spinner("智能客服思考中..."):
        # 调用瘦客户端封装的请求方法
        res_stream = chat_stream_from_api(prompt, st.session_state["session_id"])

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        # Streamlit 渲染流式数据
        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        
        # 保存完整回复到本地状态
        if response_messages:
            full_reply = "".join(response_messages)
            st.session_state["message"].append({"role": "assistant", "content": full_reply})