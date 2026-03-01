import time
import uuid
import streamlit as st
from agent.react_agent import SmartRoutingAgent 
from utils.history_manager import history_manager

# --- 页面基础配置 ---
st.set_page_config(page_title="智扫通机器人", page_icon="🤖")
st.title("🤖 智扫通机器人智能客服")
st.caption("基于 LangChain 与意图路由的智能客服引擎")
st.divider()

# ==========================================
# 1. 状态初始化 (必须放在最前面，保证全局安全)
# ==========================================
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())[:8]
if "agent" not in st.session_state:
    st.session_state["agent"] = SmartRoutingAgent()
if "message" not in st.session_state:
    st.session_state["message"] = []

# ==========================================
# 2. 侧边栏：会话管理 (彻底解决无限刷新的 Bug)
# ==========================================
with st.sidebar:
    st.subheader("💬 会话管理")
    
    # 【核心修复 1】：将“新建会话”彻底剥离成独立按钮，绝不和 selectbox 混用
    if st.button("➕ 新建对话", use_container_width=True):
        st.session_state["session_id"] = str(uuid.uuid4())[:8]
        st.session_state["message"] = []
        st.rerun() # 强制刷新页面以清空聊天框

    st.divider()

    # 获取现有会话列表
    sessions = history_manager.list_sessions()
    
    if sessions:
        # 找到当前 session 在列表中的位置，防止下拉框乱跳
        current_index = sessions.index(st.session_state["session_id"]) if st.session_state["session_id"] in sessions else 0
        
        # 【核心修复 2】：下拉框纯粹只做历史记录的展示和切换
        selected_session = st.selectbox(
            "切换历史记录",
            options=sessions,
            index=current_index
        )
        
        # 只有当用户真正手动点击切换了下拉框的值，才去加载历史并刷新
        if selected_session != st.session_state["session_id"]:
            st.session_state["session_id"] = selected_session
            st.session_state["message"] = history_manager.load_history(selected_session)
            st.rerun()

        # 删除会话
        if st.button("🗑️ 删除当前会话", type="secondary", use_container_width=True):
            history_manager.delete_history(st.session_state["session_id"])
            # 删除后自动创建一个新的空会话
            st.session_state["session_id"] = str(uuid.uuid4())[:8]
            st.session_state["message"] = []
            st.success("会话已删除")
            time.sleep(0.5) # 给用户半秒钟看绿色的成功提示
            st.rerun()
    else:
        st.info("暂无历史会话记录")
        
    # 在最下方显示一下当前的 ID，方便你调试追踪
    st.caption(f"当前 Session: `{st.session_state['session_id']}`")

# ==========================================
# 3. 主界面：渲染聊天历史记录
# ==========================================
for msg in st.session_state["message"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ==========================================
# 4. 核心交互：聊天输入与流式响应
# ==========================================
prompt = st.chat_input("请问有什么可以帮您？（例如：生成使用报告 / 小户型推荐）")

if prompt:
    # 1. 渲染用户输入，并存入前端记忆
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    # 2. 渲染 AI 响应
    with st.chat_message("assistant"):
        with st.spinner("智能客服思考中..."):
            
            # 获取后端的流式数据 (Generator)
            res_stream = st.session_state["agent"].execute_stream(prompt, st.session_state["session_id"])

            # 【核心修复 3】：用最极简的方式包装打字机效果
            def type_writer(stream):
                for chunk in stream:
                    for char in chunk:
                        time.sleep(0.015) # 稍微快一点点，体验更好
                        yield char

            # st.write_stream 会自动把 generator 里的字一个一个敲出来
            # 并且！它执行完后，会自动把完整的字符串返回给 full_response
            full_response = st.write_stream(type_writer(res_stream))

    # 3. 直接将完整的回答存入状态，再也不会只存最后一个字了
    st.session_state["message"].append({"role": "assistant", "content": full_response})