import main
from main import query_doc
from main import rerank
from gemini import classify
from milvus import Milvus
import json
import numpy as np

# Voyage+Milvus "元春和迎春的关系如何" 比原生Voyage效果好   Voyage 元迎 > 原生 迎元 > JINA 元惜 > chroma Rerank Cohere 
topic = ["红楼梦", 'hawaii', 'florida']
querys = [ 'This is a query document about florida',
    "谁和黛玉关系最好", 
    "一张弓，弓上挂着一个香橼。说的是谁", 
    "元春和迎春的关系如何",
]
query = querys[3]

# 1 test Voyage
query_doc(topic[0] + query, 2, True, 20) # local call
# query_doc() # call by remote client
rerank(topic[0] + query, 2)

# 2 test Milvus
# from pymilvus.model.dense import JinaEmbeddingFunction # 768
# jina_fn = JinaEmbeddingFunction(
#     model_name="jina-embeddings-v2-base-en", # Defaults to `jina-embeddings-v2-base-en`
#     api_key="jina_4129a0d4fdd9469785d8a9728c6f4d9fUGPF0NemmXI_uVRHvnfGLImuEoyq"
# )
# milvus = Milvus(jina_fn)
milvus = Milvus()
# 重置数据
milvus.create_db("literature", 1024)
milvus.upsert_docs("literature", main.get_my_documents(), "criticism")
# milvus.create_db("questions", 1024)
# milvus.upsert_docs("questions", query, "literature")
# 查询 by id
# data = milvus.get_by_ids("literature", [
#     0,1,2
# ], ["text", "subject", "metadata"]) # "vector", 
# print(data)
print(milvus.search("literature", query, 2)) # 查1个问题 query , 查多个问题 ["text", "subject"]

# 3 test gemini
# classify("```json{ \"topic\": \"红楼梦\", \"input_text\": \"林黛玉\", \"data\": \"飞机 \"}```")
# classify("```json{ \"topic\": \"红楼梦\", \"input_text\": \"林黛玉\", \"data\": \"贾宝玉 \"}```")