import os
import requests
from dotenv import load_dotenv

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")
BASE_URL = "https://serpapi.com/search.json"


def serpapi_search(query: str, engine: str = "google", num_results: int = 5):
    """
    Generic SerpAPI search wrapper
    """
    params = {
        "q": query,
        "engine": engine,
        "api_key": SERPAPI_KEY,
        "num": num_results,
    }

    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    return response.json()


def get_trending_topics():
    """
    Fetch trending topics using Google Trends via SerpAPI
    """
    params = {
        "engine": "google_trends_trending_now",
        "geo": "US",
        "api_key": SERPAPI_KEY,
    }

    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    return response.json()
