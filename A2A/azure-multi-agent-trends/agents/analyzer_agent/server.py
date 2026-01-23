from fastapi import FastAPI
from agents.analyzer_agent.agent import analyze_trend

app = FastAPI(title="Trend Analyzer Agent")


@app.get("/analyze")
def analyze(topic: str):
    return analyze_trend(topic)
