from abc import ABC, abstractmethod
from typing import Optional
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_ollama import ChatOllama 
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from utils.config_handler import rag_config

class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass

class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        # 1. 初始化主模型（阿里千问线上 API）
        primary_model = ChatTongyi(model=rag_config["chat_model_name"])  # type: ignore
        
        # 2. 初始化备用容灾模型为本地 Ollama
        # 假设你在本地运行了 qwen2.5:7b 或者 llama3，请将 "qwen2.5:7b" 替换为你本地真实 pull 下来的模型名
        # temperature 设置尽量与主模型保持一致，以保证用户体验稳定
        fallback_model = ChatOllama(
            model="qwen3:8b",  # 请修改为你本地实际的模型名称，如 llama3, qwen:7b 等
            temperature=0.7,
            # 如果你的 Ollama 不在默认的 11434 端口或部署在其他服务器，可以加上 base_url 参数
            # base_url="http://127.0.0.1:11434" 
        )
        
        # 3. 使用 with_fallbacks 组装高可用模型链路
        # 当 primary_model (千问) 抛出异常或超时，自动降级路由到 fallback_model (本地 Ollama)
        robust_model = primary_model.with_fallbacks([fallback_model])
        
        return robust_model # type: ignore

class EmbeddingFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings]:
        return DashScopeEmbeddings(model=rag_config["embedding_model_name"])  # type: ignore
    
chat_model = ChatModelFactory().generator()
embed_model = EmbeddingFactory().generator()