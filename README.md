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
## Setup
1. Clone the repository
2. Set up environment variables
    ```
    # 1 setup safety password
    touch .env
    # 2 setup project env
    sudo apt install python3.12-venv -y
    python3 -m venv .venv
    # 3 use venv
    source .venv/bin/activate
    ```
3. Install dependencies: `pip install -r requirements.txt`
4. Run the application `python voyageai-hello-world.py`
## Milvus
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