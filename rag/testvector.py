import sys
import os 
import logging

# --- 1. 核心路径修复 ---
current_path = os.path.abspath(__file__)
rag_dir = os.path.dirname(current_path)
project_root = os.path.dirname(rag_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 禁用所有代理环境变量，避免梯子干扰本地服务或 SSL 校验
os.environ['no_proxy'] = '*'
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

from langchain_chroma import Chroma
from utils.config_handler import chroma_config
from model.factory import chat_model, embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.logger_handle import logger
from utils.file_handler import PDf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from rag.ensembletools import RRFRanker, HybridRetriever
from utils.db_handler import db_manager

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

class VectorStoreService(object):
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_config["collection_name"],
            embedding_function=embed_model,
            persist_directory=chroma_config["persist_directory"],
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chunk_size"],
            chunk_overlap=chroma_config["chunk_overlap"],
            length_function=len,
            separators=chroma_config["separator"],
        )

        # 初始化 BM25 检索器
        self.bm25_retriever = None
        self._init_bm25_from_chroma()

    def _init_bm25_from_chroma(self):
        """从现有的 Chroma 数据库中提取所有文本块，构建 BM25 词频索引"""
        try:
            all_data = self.vector_store.get()
            doc_texts = all_data.get("documents", [])
            
            if doc_texts:
                documents = [Document(page_content=text) for text in doc_texts]
                # 注意：此步依赖 rank_bm25 包
                self.bm25_retriever = BM25Retriever.from_documents(documents)
                self.bm25_retriever.k = chroma_config.get("k", 5)
                logger.info(f"[BM25] 混合检索器词频索引构建成功，共加载 {len(documents)} 个文本块。")
            else:
                logger.warning("[BM25] Chroma 库为空，暂不初始化 BM25 检索器。")
        except Exception as e:
            logger.error(f"[BM25] 初始化失败: {e}", exc_info=True)

    def get_retriever(self, search_type="mmr"):
        """
        组装检索器，支持对比不同的向量搜索模式
        :param search_type: "mmr" (侧重多样性，去重) 或 "similarity" (纯相似度，最快)
        """
        
        k_value = chroma_config.get("k", 5)

        # 1. 根据参数配置向量检索路径
        if search_type == "mmr":
            # MMR 模式：从 3倍 的候选结果中筛选出最不重复的 k 条
            vector_retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k_value,
                    "fetch_k": k_value * 3,  
                    "lambda_mult": 0.5       # 0.5 是相关性与多样性的平衡点
                }
            )
        else:
            # Similarity 模式：直接返回相似度最高的 k 条
            vector_retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k_value}
            )

        # 2. 如果 BM25 未就绪，直接返回单路向量检索
        if not self.bm25_retriever:
            logger.warning(f"[检索器] BM25未就绪，使用单路向量({search_type})检索")
            return vector_retriever

        # 3. 使用 RRF 算法组装混合检索
        # 即使向量模式不同，我们依然可以把它们与 BM25 (关键词) 进行融合
        logger.info(f"[混合检索] 正在组装：BM25 + 向量({search_type})")
        rrf_ranker = RRFRanker(c=60, top_k=k_value)
        
        return HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=self.bm25_retriever,
            rrf_ranker=rrf_ranker
        )
    def load_document(self):
        """加载文档入库并计算 MD5 过滤重复文件"""
        
        # 1. 内部函数：检查数据库中是否存在该 MD5
        def check_md5_hex(md5_for_check: str):
            return db_manager.is_file_exists(md5_for_check)

        # 2. 内部函数：将新文件记录写入数据库
        def save_md5_hex(file_path: str, md5_for_save: str):
            file_name = os.path.basename(file_path)
            # 注意：这里调用的是 db_handler.py 里的 add_file_record 方法
            db_manager.add_file_record(file_name, md5_for_save)

        # 3. 内部函数：根据文件类型选择加载器
        def get_file_documents(file_path: str):
            if file_path.endswith(".pdf"): return PDf_loader(file_path)
            if file_path.endswith(".txt"): return txt_loader(file_path)
            return []

        # 4. 获取目录下允许的文件列表
        allowed_type_files = listdir_with_allowed_type(
            chroma_config["data_path"], 
            tuple(chroma_config["allow_knowledge_file_type"])
        )

        is_new_doc_added = False

        for path in allowed_type_files:
            md5_hex = get_file_md5_hex(path)
            if md5_hex is None:
                logger.error(f"[MD5] 文件 {path} 计算失败")
                continue

            # --- 步骤 A: 查库判重 ---
            if check_md5_hex(md5_hex):
                logger.warning(f"[MD5] 文件 {path} 已存在于 MySQL，跳过")
                continue

            try: 
                # --- 步骤 B: 加载与切割 ---
                documents = get_file_documents(path)
                if not documents: continue

                split_documents = self.spliter.split_documents(documents)
                if not split_documents: continue

                # --- 步骤 C: 存入向量库 ---
                self.vector_store.add_documents(split_documents)
                
                # --- 步骤 D: 存入 MySQL (这里修正了参数) ---
                # 错误写法: save_md5_hex(md5_hex) 
                # 正确写法: ↓↓↓
                save_md5_hex(path, md5_hex) 
                
                logger.info(f"[加载知识库] 文件 {path} 添加成功并记录到 MySQL")
                is_new_doc_added = True

            except Exception as e:
                logger.error(f"[加载知识库] 失败: {path}, 错误: {e}", exc_info=True)
                continue

        # 如果有新内容，刷新内存中的 BM25 索引
        if is_new_doc_added:
            logger.info("[BM25] 检测到新文档入库，正在刷新内存索引...")
            self._init_bm25_from_chroma()
            
if __name__ == "__main__":
    service = VectorStoreService()
    service.load_document()
    
    test_query = "机器人迷路了怎么办"
    
    # 对比测试：分别查看两种模式下的融合结果
    for mode in ["similarity", "mmr"]:
        print(f"\n\n{'='*30}")
        print(f" 正在测试模式: {mode.upper()} ")
        print(f"{'='*30}")
        
        # 获取对应模式的检索器
        retriever = service.get_retriever(search_type=mode)
        
        # 执行检索
        results = retriever.invoke(test_query)
        
        # 打印结果
        for idx, doc in enumerate(results): 
            print(f"[{mode} 结果 {idx+1}] {doc.page_content[:60].replace(os.linesep, '')}...")