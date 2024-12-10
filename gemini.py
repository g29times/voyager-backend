import google.generativeai as genai
import os

def classify(text):
    # genai.configure(api_key=os.environ["API_KEY"])
    genai.configure(api_key="AIzaSyCMxfYl6Vo4EkatPNNEcorlcAtG4tBHccY")

    model = genai.GenerativeModel(model_name="gemini-1.5-flash", system_instruction="""
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
    """)
    response = model.generate_content(text)
    print(response.text)
    return f"{response.text}"