import voyageai
import os

vo = voyageai.Client(api_key=os.getenv('VOYAGE_API_KEY'))
# This will automatically use the environment variable VOYAGE_API_KEY.
# Alternatively, you can use vo = voyageai.Client(api_key="<your secret key>")

result = vo.embed(["hello world"], model="voyage-3")
print(result)