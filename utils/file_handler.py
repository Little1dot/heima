import os
import hashlib
from utils.logger_handle import logger
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document

# 获取文件md5
def get_file_md5_hex(filepath: str):
    if not os.path.exists(filepath):  # 判断文件是否存在
        logger.error(f"[md5计算]文件{filepath}不存在")
        return 
    
    if not os.path.isfile(filepath):  # 判断是否为文件
        logger.error(f"[md5计算]文件{filepath}不是文件")
        return
    
    # 创建md5对象
    md5_obj = hashlib.md5()
    # 大文件读取方式
    chunk_size = 4096 # 避免文件过大
    try: 
        with open(filepath, 'rb') as f:   # rb为只读模式，必须为二进制
            while chunk:= f.read(chunk_size): 
                # 使用了海象运算符（:=），在读取文件时循环处理数据块。
                # 具体功能是：每次从文件 f 中读取大小为 chunk_size 的数据块，并赋值给变量 chunk，只要 chunk 不为空就继续循环。
                md5_obj.update(chunk) 
        return md5_obj.hexdigest()
    except Exception as e:
        logger.error(f"[md5计算]文件{filepath}计算md5失败,错误信息为:{e}")
        return None

# 读取允许的类型的文件，用于判断哪些文件允许读取，并且整理为列表
def listdir_with_allowed_type(path: str, allowed_type: tuple[str]):
    files = []

    if not os.path.isdir(path):
        logger.error(f"[文件读取][listdir_with_allowde_type]{path}不是文件夹")
        return allowed_type
    # 获取文件夹下的所有文件，允许读取的文件列表
    for f in os.listdir(path):
        if f.endswith(allowed_type): # 判断文件类型
            files.append(os.path.join(path, f))
    return tuple(files)


# pdf加载器
def PDf_loader(filepath: str, password= None) -> list[Document]:
    return PyPDFLoader(filepath, password).load()

# txt加载器
def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath, encoding='utf-8').load()
