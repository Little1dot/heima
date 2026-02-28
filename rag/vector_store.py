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

from langchain_chroma import Chroma
from utils.config_handler import chroma_config
from model.factory import chat_model, embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.logger_handle import logger
from utils.file_handler import PDf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
import logging
from langchain.retrievers.multi_query import MultiQueryRetriever


class VectorStoreService(object):
    def __init__(self):
        # chroma数据库的配置在chroma.yml中存储，需要通过config_handler文件中的chroma.cOnfig()方法读取
        self.vector_store = Chroma(
            collection_name=chroma_config["collection_name"],
            embedding_function=embed_model,
            persist_directory=chroma_config["persist_directory"],
        )

        self.spliter =RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chunk_size"],
            chunk_overlap=chroma_config["chunk_overlap"],
            length_function=len,
            separators=chroma_config["separator"],
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_config["k"]})

    def load_document(self):
        """
        读取文件并存储入向量库
        并且计算md5，防止重复上传
        """
        def check_md5_hex(md5_for_check: str):
            # 相对路径，要转化为绝对路径
            if not os.path.exists(get_abs_path(chroma_config["md5_hex_store"])):
                open(get_abs_path(chroma_config["md5_hex_store"]), 'w', encoding='utf-8') 
                return False
            
            with open(get_abs_path(chroma_config["md5_hex_store"]), 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line  = line.strip()
                    if line == md5_for_check:
                        return True

                return False
        def save_md5_hex(md5_for_save: str):
            with open(get_abs_path(chroma_config["md5_hex_store"]), 'a', encoding='utf-8') as f:
                f.write(md5_for_save + '\n')

    # 加载文档到数据库，需要读文件转为document
        def get_file_documents(file_path: str):
            # endswith判断文件格式符合则返回true
            if file_path.endswith(".pdf"):
                return PDf_loader(file_path)
            elif file_path.endswith(".txt"):
                return txt_loader(file_path)
            else:
                return []

        
        allowed_type_files = listdir_with_allowed_type(chroma_config["data_path"], tuple(chroma_config["allow_knowledge_file_type"]))

        for path in allowed_type_files:
            md5_hex = get_file_md5_hex(path)

            if  md5_hex is None:
                logger.error(f"[md5计算]文件{path}计算md5失败")
                continue

            if check_md5_hex(md5_hex):
                logger.warning(f"[md5计算]文件{path}已存在")
                continue
            try: 
                documents = get_file_documents(path)
                if  not documents: 
                    logger.warning(f"[加载知识库]文件{path}没有有效内容，跳过")
                    continue

                split_documents = self.spliter.split_documents(documents)

                if not split_documents: 
                    logger.warning(f"[加载知识库]分割文件{path}没有有效内容，跳过")
                    continue

                # 添加到向量库
                self.vector_store.add_documents(split_documents)

                # 保存md5
                save_md5_hex(md5_hex)
                logger.info(f"[加载知识库]文件{path}添加成功")
            except Exception as e:
                # exc_info=True打印详细的报错堆栈,如果为false,则只打印错误信息本身
                logger.error(f"[加载知识库]文件{path}添加失败: {e}", exc_info=True)
                continue

if __name__ == "__main__":
    vector_store = VectorStoreService()

    vector_store.load_document()

    retriever = vector_store.get_retriever()

    res = retriever.invoke("迷路")

    for i in res: 
        print("-"*20)
        print(i.page_content)
        print("-"*20)