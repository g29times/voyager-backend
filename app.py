# app.py
from flask import Flask, json, jsonify, request
from main import query_doc  # 从main模块导入 query_doc 函数
from gemini import classify
from milvus import Milvus

app = Flask(__name__)

# 创建数据库连接的单例实例
# db_client = MilvusClientSingleton.get_instance()

# 向量检索 使用curl构建GET请求进行测试
# curl "http://localhost:5000/search"
# 红楼梦 元春和迎春 curl "http://localhost:5000/search?topic=%E7%BA%A2%E6%A5%BC%E6%A2%A6&query=%E5%85%83%E6%98%A5%E5%92%8C%E8%BF%8E%E6%98%A5&k=3"
@app.route('/search', methods=['GET'])
def search_api():
    topic = request.args.get('topic', '红楼梦')  # 从请求的查询参数中获取topic
    query = request.args.get('query', '')  # 从请求的查询参数中获取query
    k = int(request.args.get('k', 1))  # 从请求的查询参数中获取k，如果没有提供则默认为1
    print(f"topic: {topic}, query: {query}, k: {k}")

    # 调用main.py中的 query_doc 函数
    # result_json = query_doc(topic + ", " + query, k)
    result_json = Milvus().search(topic + ", " + query, k)
    # print(result_json)
    response_data = json.dumps(result_json, ensure_ascii=False)
    # 返回JSON格式的响应
    return response_data

# Gemini分类 使用curl构建GET请求进行测试
# curl "http://localhost:5000/classify"
# curl "http://localhost:5000/classify?text=%60%60%60json%7B%20%5C%22topic%5C%22%3A%20%5C%22%E7%BA%A2%E6%A5%BC%E6%A2%A6%5C%22%2C%20%5C%22input_text%5C%22%3A%20%5C%22%E8%B4%BE%E5%AE%9D%E7%8E%89%5C%22%2C%20%5C%22data%5C%22%3A%20%5C%22%E5%92%8C%E5%B0%9A%20%5C%22%7D%60%60%60"
@app.route('/classify', methods=['GET'])
def classify_api():
    text = request.args.get('text', '```json{ \"topic\": \"红楼梦\", \"input_text\": \"林黛玉\", \"data\": \"贾宝玉 \"}```')
    response = classify(text)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)