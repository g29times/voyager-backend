import rag.voyager
from llm.gemini import classify
from db.milvus import Milvus
import json
import numpy as np

# 测试问题："元春和迎春的关系如何"
# 测试结论：Voyage+Milvus 比原生Voyage效果好
# 效果排序：1 Voyage+Milvus：元 迎 > 2 原生Voyage：迎 元 > 3 JINA：元 惜 > 
# 4 chroma + Cohere（Rerank）？ > 5 Voyage Rerank 刘心武
topics = ["红楼梦", "女儿国王"]
querys = [
    "谁和黛玉关系最好", 
    "一张弓，弓上挂着一个香橼。说的是谁", 
    "元春和迎春的关系如何", # query 2 
    "江辉工作顺利吗?" # 3
]
topic = topics[0]
query = querys[2]
top_k = 4
voyageai = rag.voyager
# 1 test Voyage # 与milvus_demo结论有不同 因这里（掉接口）使用knn_algo，而milvus_demo使用了cosine similarity
print("Test Voyage Start ---------------- ")
voyageai.query_doc(topic + query, top_k, True, 20) # local call
# query_doc() # call by remote client
voyageai.rerank(topic + query, top_k)
print("Test Voyage Finish ---------------- ")


# 2 test Milvus # TODO 与milvus_demo结论有不同 需调查原因
print("Test Milvus Start ---------------- ")
from pymilvus.model.dense import JinaEmbeddingFunction
jina_fn = JinaEmbeddingFunction(
    model_name="jina-embeddings-v3",
    api_key="jina_4129a0d4fdd9469785d8a9728c6f4d9fUGPF0NemmXI_uVRHvnfGLImuEoyq"
)
milvus = Milvus(jina_fn)
# milvus = Milvus() # milvus origin embedding
# 重置数据
milvus.create_db("literature", 1024) # dimension see db/milvus_demo.py
milvus.upsert_docs("literature", voyageai.get_my_documents(), "文学评论", "曹雪芹")
# milvus.create_db("questions", 1024)
# milvus.upsert_docs("questions", query, "literature")
# 查询 by id
# data = milvus.get_by_ids("literature", [
#     0,1,2
# ], ["text", "subject", "metadata"]) # "vector", 
# print(data)
print(milvus.search("literature", query, top_k)) # 查1个问题 query , 查多个问题 ["text", "subject"]

# 3 test gemini
print("Test GEMINI Start ---------------- ")
# classify("```json{ \"topic\": \"红楼梦\", \"input_text\": \"林黛玉\", \"data\": \"飞机 \"}```")
# classify("```json{ \"topic\": \"红楼梦\", \"input_text\": \"林黛玉\", \"data\": \"贾宝玉 \"}```")