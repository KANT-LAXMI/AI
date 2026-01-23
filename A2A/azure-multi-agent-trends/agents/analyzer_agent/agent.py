from common.azure_openai_client import chat_completion
from common.serpapi_search import serpapi_search
import json


SYSTEM_PROMPT = """
You are a data analyst.
Focus strictly on quantitative metrics.
"""


def analyze_trend(topic: str):
    search_data = serpapi_search(f"{topic} social media statistics")

    context = json.dumps(search_data)[:4000]

    user_prompt = f"""
Analyze this trend with numbers only:
- mentions
- growth rate
- geography
- hashtags

Trend: {topic}

DATA:
{context}
"""

    return chat_completion(SYSTEM_PROMPT, user_prompt)
