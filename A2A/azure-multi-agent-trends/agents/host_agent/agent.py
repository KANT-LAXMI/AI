import requests


TRENDING_AGENT_URL = "http://localhost:10020/find_trends"
ANALYZER_AGENT_URL = "http://localhost:10021/analyze"


def run_full_analysis():
    trends_response = requests.get(TRENDING_AGENT_URL).json()

    top_trend = trends_response["trends"][0]["topic"]

    analysis_response = requests.get(
        ANALYZER_AGENT_URL,
        params={"topic": top_trend}
    ).json()

    return {
        "top_trend": top_trend,
        "analysis": analysis_response
    }
