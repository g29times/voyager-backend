import voyager
from gemini import classify
from milvus import Milvus
import json
import numpy as np

# 测试问题："元春和迎春的关系如何"
# 测试结论：Voyage+Milvus 比原生Voyage效果好
# 效果排序：Voyage+Milvus：元 迎 > 原生Voyage：迎 元 > JINA：元 惜 > chroma + Cohere（Rerank）？ > Voyage Rerank 刘心武
topic = ["红楼梦", "女儿国王"]
querys = [
    "谁和黛玉关系最好", 
    "一张弓，弓上挂着一个香橼。说的是谁", 
    "元春和迎春的关系如何", # query 2 
    "江辉工作顺利吗?" # 3
]
query = querys[2]
topic = topic[0]
top_k = 3

# 1 test Voyage
print("Test Voyage Start ---------------- ")
voyager.query_doc(topic[0] + query, top_k, True, 20) # local call
# query_doc() # call by remote client
voyager.rerank(topic[0] + query, top_k)
print("Test Voyage Finish ---------------- ")


# 2 test Milvus
print("Test Milvus Start ---------------- ")
# from pymilvus.model.dense import JinaEmbeddingFunction # 768
# jina_fn = JinaEmbeddingFunction(
#     model_name="jina-embeddings-v2-base-en", # Defaults to `jina-embeddings-v2-base-en`
#     api_key="jina_4129a0d4fdd9469785d8a9728c6f4d9fUGPF0NemmXI_uVRHvnfGLImuEoyq"
# )
# milvus = Milvus(jina_fn)
# from pymilvus.model.hybrid import BGEM3EmbeddingFunction # local model size 8G speed slow
# embedding_fn = BGEM3EmbeddingFunction(
#     model_name='BAAI/bge-m3', # Specify the model name
#     device='cpu', # Specify the device to use, e.g., 'cpu' or 'cuda:0'
#     use_fp16=False # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
# )
# milvus = Milvus(embedding_fn)
milvus = Milvus() # milvus origin embedding
# 重置数据
milvus.create_db("literature", 1024)
milvus.upsert_docs("literature", voyager.get_my_documents(), "criticism")
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