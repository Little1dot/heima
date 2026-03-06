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
from numpy.linalg import norm
from pydantic import BaseModel, Field
from typing import Literal
from utils.logger_handle import logger
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model, embed_model
from utils.prompt_loader import load_classify_prompt

# ==========================================
# 1. 结构化输出 Schema (慢速通道的数据契约)
# ==========================================
class RouteDecision(BaseModel):
    """用于决定用户意图的路由分类器"""
    intent: Literal["chat", "rag", "task"] = Field(
        ..., 
        description="chat: 日常打招呼与闲聊; rag: 关于扫地机器人的知识、故障排查、选购等问题; task: 查天气、定位或生成报告等需要调用工具的复杂指令。"
    )

class QueryRewriter:
    """
    用于多轮对话的指代消解与上下文补全
    把形如 "它有什么缺点" 结合上下文重写为 "T10扫地机有什么缺点"
    """
    def __init__(self):
        self.rewrite_prompt = PromptTemplate.from_template(
            "你是一个专业的搜索词优化专家。请根据下方的【历史对话上下文】，将用户的【最新问题】重写为一个独立、完整、意图明确的句子。\n"
            "【规则】：\n"
            "1. 补全最新问题中缺失的主语、代词（如把'它'替换为具体的设备型号）。\n"
            "2. 如果最新问题已经很完整，或者与历史对话无关（如开始了新话题），请原样输出。\n"
            "3. 绝对不要回答问题，只输出重写后的句子！\n\n"
            "【历史对话上下文】：\n"
            "{history}\n\n"
            "【用户最新问题】：{query}\n\n"
            "重写后的独立问题："
        )

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
        
# ==========================================
# 2. 语义路由器 (极速通道：空间换时间)
# ==========================================
class SemanticRouter:
    def __init__(self):
        # 1. 准备标准意图的高频语料库
        # 这里的语料可以根据你们系统实际运行后收集到的高频 User Query 不断扩充
        self.routes = {
            "chat": [
                "你好", "在吗", "你是谁", "早上好", "很高兴认识你", 
                "你能做什么", "哈哈", "谢谢"
            ],
            "rag": [
                "怎么清理尘盒", "机器报红灯05怎么修", "说明书在哪", 
                "小户型选哪个型号好", "扫地机充不进电怎么办", "主刷被头发缠住了"
            ],
            "task": [
                "帮我查一下天气", "今天北京会下雨吗", "生成我的专属报告", 
                "看看我这个月扫了多少次", "定位一下我现在的位置"
            ]
        }
        self.embeddings_cache = {}
        # 2. 启动时自动进行向量化预热
        self._warmup()

    def _warmup(self):
        """系统启动时，将预设语料提前向量化并缓存在内存中"""
        print("[启动加载] 正在预热 Semantic Router 向量缓存...")
        for intent, examples in self.routes.items():
            # 调用你系统里的 DashScopeEmbeddings 批量转向量
            vectors = embed_model.embed_documents(examples)
            self.embeddings_cache[intent] = vectors
        print("[启动加载] Semantic Router 预热完成！")

    def _cosine_similarity(self, vec1, vec2):
        """计算两个向量的余弦相似度"""
        return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

    def fast_route(self, query: str, threshold: float = 0.82) -> str:
        """极速路由检测"""
        # 将用户的输入转为向量
        query_vec = embed_model.embed_query(query)
        
        max_score = 0
        best_intent = None
        
        # 遍历比对，寻找最相似的意图语料
        for intent, vectors in self.embeddings_cache.items():
            for vec in vectors:
                score = self._cosine_similarity(query_vec, vec)
                if score > max_score:
                    max_score = score
                    best_intent = intent
                    
        # 如果最高相似度超过设定的安全阈值，直接判定意图并返回
        if max_score >= threshold:
            print(f"[极速路由命中] 匹配意图: [{best_intent}], 置信度: {max_score:.4f}")
            return best_intent
            
        print(f"[极速路由未命中] 最高置信度 {max_score:.4f} 低于阈值 {threshold}。")
        return None 

# ==========================================
# 3. 混合路由器 (终极对外暴露的类)
# ==========================================
class HybridIntentRouter:
    """混合意图路由器：先尝试极速向量匹配，匹配失败再走 LLM 深度推理"""
    def __init__(self):
        # 1. 初始化一级极速路由
        self.semantic_router = SemanticRouter()
        
        # 2. 初始化二级 LLM 深度推理路由
        self.llm_router_chain = self._build_llm_router()

    def _build_llm_router(self):
        """构建带有结构化约束的大模型路由链"""
        prompt_text = load_classify_prompt() 
        router_prompt = PromptTemplate.from_template(prompt_text)
        
        # 绑定 Pydantic 约束
        structured_llm = chat_model.with_structured_output(RouteDecision)
        
        return router_prompt | structured_llm

    def predict_intent(self, query: str) -> str:
        """
        对外暴露的核心预测方法
        返回: "chat", "rag", 或 "task"
        """
        # --- 第一关：极速通道 (Fast Path) ---
        # 能够拦截 80% 以上的高频常见问法，耗时 ~50ms
        intent = self.semantic_router.fast_route(query, threshold=0.85)
        if intent:
            return intent
            
        # --- 第二关：慢速深度思考通道 (Slow Path) ---
        # 处理长尾问题或复杂指令，耗时 ~500ms+
        print("[路由更改] 启动 LLM 结构化输出进行深度意图推理...")
        try:
            decision: RouteDecision = self.llm_router_chain.invoke({"query": query})
            print(f"[深度推理命中] 解析意图: [{decision.intent}]")
            return decision.intent
        except Exception as e:
            # 终极物理兜底：如果 API 崩溃或极度异常，默认走最保守的 RAG 或者 Chat
            print(f"[路由异常] LLM 意图解析彻底失败，触发物理兜底。错误信息: {e}")
            return "chat"
        
        
query_rewriter = QueryRewriter()

if __name__ == "__main__":
    import time
    
    print("="*50)
    print("🚀 开始初始化混合路由器 (包含向量预热)...")
    print("="*50)
    
    # 实例化路由器，会自动触发 SemanticRouter 的 _warmup()
    router = HybridIntentRouter()
    
    print("\n" + "="*50)
    print("🎯 开始测试 Query 路由分发")
    print("="*50)
    
    # 精心设计的测试用例库
    test_queries = [
        # --- 测试组 1：高频简单问题（预期：秒级触发极速路由） ---
        "你好啊",                                   # 预期: chat
        "怎么清理尘盒",                             # 预期: rag
        "今天北京会下雨吗",                         # 预期: task
        
        # --- 测试组 2：长尾复杂问题（预期：极速路由未命中，降级触发 LLM 深度推理） ---
        "我刚才出门不小心把水杯碰倒了，机器扫过去之后一直发出咔咔的异响，这种情况怎么修？",  # 预期: rag
        "周末家里要来客人，你能帮我查下明天的天气状况，顺便生成一份我最近的清洁报告吗？",  # 预期: task
        "你觉得扫地机器人有一天会统治人类吗？"                                              # 预期: chat
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[测试用例 {i}] ".ljust(50, '-'))
        print(f"👤 用户输入: {query}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 调用混合路由器核心方法
        intent = router.predict_intent(query)
        
        # 记录结束时间
        cost_time = time.time() - start_time
        
        print(f"✅ 最终路由: => [{intent.upper()}] <=")
        print(f"⏱️ 耗时统计: {cost_time:.4f} 秒")