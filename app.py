import time
import uuid
import streamlit as st
from agent.react_agent import SmartRoutingAgent
from utils.history_manager import history_manager

# 标题
st.title("智扫通机器人智能客服")
st.divider()

# 会话管理
with st.sidebar:
    st.subheader("会话管理")
    
    # 获取现有会话列表
    sessions = history_manager.list_sessions()
    
    # 会话选择
    selected_session = st.selectbox(
        "选择会话",
        options=["新建会话"] + sessions,
        key="session_selector"
    )
    
    # 如果选择新建会话
    if selected_session == "新建会话":
        # 生成新会话ID
        new_session_id = str(uuid.uuid4())[:8]
        st.session_state["session_id"] = new_session_id
        st.session_state["message"] = []
        st.info(f"已创建新会话: {new_session_id}")
    else:
        # 如果选择现有会话
        if "session_id" not in st.session_state or st.session_state["session_id"] != selected_session:
            st.session_state["session_id"] = selected_session
            # 加载历史消息
            st.session_state["message"] = history_manager.load_history(selected_session)
            st.info(f"已加载会话: {selected_session}")
    
    # 删除会话按钮
    if st.button("删除当前会话"):
        if "session_id" in st.session_state:
            history_manager.delete_history(st.session_state["session_id"])
            st.session_state["session_id"] = "default"
            st.session_state["message"] = []
            st.success("会话已删除")
            st.rerun()

# 确保session_id存在
if "session_id" not in st.session_state:
    st.session_state["session_id"] = "default"

if "agent" not in st.session_state:
    st.session_state["agent"] = SmartRoutingAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    response_messages = []
    with st.spinner("智能客服思考中..."):
        # 传递session_id给execute_stream方法
        res_stream = st.session_state["agent"].execute_stream(prompt, st.session_state["session_id"])

        def capture(generator, cache_list):

            for chunk in generator:
                cache_list.append(chunk)

                for char in chunk:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        st.session_state["message"].append({"role": "assistant", "content": response_messages[-1]})
        st.rerun()
