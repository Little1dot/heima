"""
rag.rag_summerize 的 Docstring
总结服务，用户提问，将搜索资料和提问提交给模型，让模型总结服务
"""
# import sys
# import os

# # --- 核心修复代码开始 ---
# # 1. 获取当前文件(vector_store.py)的绝对路径
# current_path = os.path.abspath(__file__)
# # 2. 获取父目录(rag)
# rag_dir = os.path.dirname(current_path)
# # 3. 获取爷目录(HeiMaAgent1.22)，也就是项目的根目录
# project_root = os.path.dirname(rag_dir)

# # 4. 把根目录加入到 Python 的搜索路径中 (插在第一个位置)
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompt
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

def print_prompt(prompt):
    print("-" * 20)
    print(prompt.to_string())
    print("-" * 20)
    return prompt


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self_prompt_text = load_rag_prompt()
        self.prompt_template = PromptTemplate.from_template(self_prompt_text)
        self.model = chat_model
        self.chain = self.__init__chain()

    def __init__chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain
    
    def retriever_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)
    

    def rag_summarize(self, query: str) -> str:

        docs = self.retriever_docs(query)

        context = ""
        counter = 0

        for doc in docs:
            counter += 1
            context += f"文档片段{counter}：{doc.page_content} | 文档元数据:{doc.metadata}\n"
        return self.chain.invoke(
            {"input":  query, 
            "context" : context}
            )

if __name__ == "__main__":
    rag_summarize = RagSummarizeService()
    print(rag_summarize.rag_summarize("小户型适合哪种扫地机器人"))