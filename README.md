### 项目介绍：
 本项目来自于b站黑马agent教程，因为教程中的内容较为基础。因此，我在它的基础上增加了一些内容。 
#### 整体架构如下：
```bash 
agent： query输入——>rewrite ——> router ——> 分流到不同的链路（chat/rag/task）
rag部分：优化检索逻辑，将单一向量相似度检索改写为混合检索，并手写了RRF融合器，将chromaDB改为了Milvus/Faiss，通过混合检索提高检索精度，加入rerank提高召回相关度
存储和持久化：加入mysql和redis，实现历史对话记录和数据库的上传
history：
```
#### 环境配置
安装uv
```bash
pip install uv
```
创建虚拟环境与初始化
```bash
uv sync
.venv\Scripts\activate
```
#### 配置内容
