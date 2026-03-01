from typing import List, Dict
from langchain_core.documents import Document
import collections

class RRFRanker:
    def __init__(self, c: int = 60, top_k: int = 4):
        """
        初始化 RRF 融合器
        :param c: RRF 算法常数，默认为 60（业界标准值），用于平滑排名权重
        :param top_k: 最终返回的文档数量
        """
        self.c = c
        self.top_k = top_k

    def fuse(self, results_list: List[List[Document]]) -> List[Document]:
        """
        执行倒数秩融合算法
        :param results_list: 多个检索器返回的文档列表集合，例如 [docs_from_vec, docs_from_bm25]
        :return: 融合打分后排名前 top_k 的文档列表
        """
        # 1. 用于存储每个文档的累积得分 {doc_content: score}
        doc_scores: Dict[str, float] = collections.defaultdict(float)
        # 2. 用于存储文档内容与对象的映射，方便最后还原对象
        doc_map: Dict[str, Document] = {}

        # 遍历每个检索器的结果列表
        for docs in results_list:
            # 枚举每个文档及其在该列表中的排名 (rank 从 1 开始)
            for rank, doc in enumerate(docs, start=1):
                content = doc.page_content
                if content not in doc_map:
                    doc_map[content] = doc
                
                # 计算 RRF 分数并累加: 1 / (c + rank)
                doc_scores[content] += 1.0 / (self.c + rank)

        # 3. 根据得分从高到低排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 4. 还原为 Document 对象并截取前 top_k 个
        final_results = [doc_map[content] for content, score in sorted_docs[:self.top_k]]
        
        return final_results

# 为了让这个类能像 LangChain Retriever 一样被调用，我们给它包一层
class HybridRetriever:
    def __init__(self, vector_retriever, bm25_retriever, rrf_ranker: RRFRanker):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.ranker = rrf_ranker

    def invoke(self, query: str) -> List[Document]:
        # 并行/串行获取两路结果
        vec_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        
        # 使用 RRF 算法融合并重排
        return self.ranker.fuse([vec_docs, bm25_docs])