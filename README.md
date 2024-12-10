# Project Voyager
This is a LLM based project to help you find some idea with the help of best embedding tools

# Folder Arrangement
FrontEnd: ai-voyager  
BackEnd: voyageai(RAG) + LLM  
Database: chroma_data/Milvus

# For Users
download the source, then run
`sh start.sh`

# For Developers
pymilvus==2.4.5
milvus-lite==2.4.9 # not on windows https://github.com/milvus-io/milvus/discussions/36759
milvus-model==0.2.4

## VoyageAI
https://docs.voyageai.com/docs/api-key-and-installation
pip install -U voyageai

## Milvus Lite
https://milvus.io/docs/install-overview.md
pip install -U pymilvus
https://github.com/milvus-io/milvus-model
pip install -U pymilvus[model]