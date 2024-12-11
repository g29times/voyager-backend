from zhipuai import ZhipuAI


client = ZhipuAI(api_key="ea46161b58af635ac12fea0fd01c1910.AUdc2dLYIjIyvMoD")  # 请填写您自己的APIKey
def classify(text):
    response = client.chat.completions.create(
        model="glm-4-plus",  # 请填写您要调用的模型名称
        messages=[
            {"role": "system", "content": """
            ### Job Description
            You are a text classification engine that analyzes text data and assigns categories based on user input or automatically determined categories.

            ### Task
            Your task is to assign one categories ONLY to the input text and only one category may be assigned returned in the output.Additionally, you need to extract the key words from the text that are related to the classification.

            ### Format
            The input text is in the variable input_text.Categories are specified as a category list with two filed category_id and category_name in the variable categories .Classification instructions may be included to improve the classification accuracy.

            ### Constraint
            DO NOT include anything other than the JSON array in your response.

            ### Memory
            Here is the chat histories between human and assistant, inside <histories></histories> XML tags.
            <histories>
                [
            {
                "role": "user",
                "text": "```json{\"topic\": \"红楼梦\", \"input_text\": \"贾宝玉\", \"data\": \"林黛玉\", \"categories\": 
                [{\"category_id\":\"1\",\"category_name\":\"相关\"},{\"category_id\":\"2\",\"category_name\":\"不相关\"}],    
                \"classification_instructions\": [\"结合主题topic，判断输入input_text是否与数据data相关。\"]}```",
                "files": [

                ]
            },
            {
                "role": "assistant",
                "text": "```json{\"topic\": \"红楼梦\", \"input_text\": \"贾宝玉\", \"category_id\": \"1\", \"category_name\": \"相关\"}```",
                "files": [

                ]
            },
            {
                "role": "user",
                "text": "```json{\"topic\": \"红楼梦\", \"input_text\": \"贾宝玉\", \"data\": \"魔兽世界。。。\"}```",
                "files": [

                ]
            },
            {
                "role": "assistant",
                "text": "```json{\"topic\": \"红楼梦\", \"input_text\": \"贾宝玉\", \"category_id\": \"2\",  \"category_name\": \"不相关\"}```",
                "files": [

                ]
            }
        ]
            </histories>
            """},
            {"role": "user", "content": text},
        ],
        # stream=True,
    )
    messages = response.choices[0].message
    print(messages)
    return messages
    # for chunk in response:
    #     print(chunk.choices[0].delta)

if __name__ == "__main__":
    classify("我对太阳系的行星非常感兴趣，尤其是土星。请提供关于土星的基本信息，包括它的大小、组成、环系统以及任何独特的天文现象。")