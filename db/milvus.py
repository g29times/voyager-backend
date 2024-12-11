from pymilvus import MilvusClient

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rag.voyager

# Docs
# 使用 https://milvus.io/api-reference/pymilvus/v2.4.x/MilvusClient/Vector/get.md
# 接入三方 https://milvus.io/docs/embed-with-voyage.md
# 原理 https://milvus.io/docs/metric.md?tab=floating
# 原理 https://milvus.io/docs/index.md?tab=sparse
# 接入三方 https://docs.llamaindex.ai/en/stable/examples/vector_stores/MilvusHybridIndexDemo/?h=milvus
# 接入三方 https://docs.zilliz.com.cn/reference/restful/query-v2?_highlight=%E6%9F%A5%E8%AF%A2

from pymilvus.model.dense import VoyageEmbeddingFunction
from pymilvus.model.dense import JinaEmbeddingFunction
from pymilvus.model.dense import CohereEmbeddingFunction

# client = MilvusClient("milvus_demo.db")

class Milvus:
    def __init__(self, embedding_fn=None, client=None):
        self.client = client if client is not None else MilvusClient("milvus_demo.db")
        if embedding_fn is None:
            embedding_fn = VoyageEmbeddingFunction(
                model_name="voyage-2", # Defaults to `voyage-2`
                api_key="pa-ReOQxAJwGywtO4bfpQVnjyJv5uHsqnBTC0ym8DE73Yg"
            )
        self.embedding_fn = embedding_fn

    # 1 Create db self.embedding_fn.dim
    def create_db(self, collection_name, dim):
        if self.client.has_collection(collection_name=collection_name):
            self.client.drop_collection(collection_name=collection_name)
        self.client.create_collection(
            collection_name=collection_name,
            dimension=dim,  # The vectors we will use in this demo has 768 dimensions
        )
        print("create_db with dimension", dim)

    # 2 Represent text
    def encode_documents(self, docs):
        docs_embeddings = self.embedding_fn.encode_documents(docs)
        print("Docs:", len(docs))
        print("Docs Dim:", self.embedding_fn.dim, docs_embeddings[0].shape)
        return docs_embeddings
    
    def encode_query(self, query):
        query_vectors = self.embedding_fn.encode_queries(query)
        print("Query Dim:", self.embedding_fn.dim, query_vectors[0].shape)
        return query_vectors

    # 3 Build Data 
    # Each entity has id, vector representation, raw text, and a subject label to filtering metadata.
    def build_data(self, docs, docs_embeddings):
        data = [
            {"id": i, "vector": docs_embeddings[i], "text": docs[i], "subject": "history"}
            for i in range(len(docs_embeddings))
        ]
        return data

    # 4 Update Insert Data
    def upsert_data(self, collection_name, data):
        res = self.client.upsert(collection_name=collection_name, data=data)
        print(res)

    # Public方法 Update Docs 聚合方法 subject可用于后续partition
    def upsert_docs(self, collection_name, docs, subject="criticism", author="Cao Xue Qin"):
        docs_embeddings = self.encode_documents(docs)
        data = [
            {"id": i, "vector": docs_embeddings[i], "text": docs[i], "subject": subject, "metadata": {"author": author}}
            for i in range(len(docs_embeddings))
        ]
        res = self.client.upsert(collection_name=collection_name, data=data)
        print(res)

    # Public方法
    def get_by_ids(self, collection_name, ids, output_fields=["text"]):
        res = self.client.get(
            collection_name=collection_name,
            output_fields=output_fields,
            ids=ids
        )
        return res

    # Public方法 Semantic Search client.search()是milvus 内部逻辑 默认使用cosine similarity
    def search(self, collection_name, query, limit=3, output_fields=["text"]):
        query_vectors = self.encode_query(query)

        res = self.client.search(
            collection_name=collection_name,  # target collection
            data=query_vectors,  # query vectors
            limit=limit,  # number of returned entities
            output_fields=output_fields,  # specifies fields to be returned
        )
        return res
