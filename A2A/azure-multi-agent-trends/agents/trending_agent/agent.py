from common.azure_openai_client import chat_completion
from common.serpapi_search import get_trending_topics
import json


SYSTEM_PROMPT = """
You are a social media trends analyst.
Return ONLY valid JSON.
"""


def find_trends():
    trends_data = get_trending_topics()
    context = json.dumps(trends_data)[:4000]

    user_prompt = f"""
From the following trending data (last 24 hours),
extract top 3 social media trends.

Return ONLY this JSON format:
{{
  "trends": [
    {{
      "topic": "",
      "description": "",
      "reason": ""
    }}
  ]
}}

DATA:
{context}
"""

    return chat_completion(SYSTEM_PROMPT, user_prompt)
