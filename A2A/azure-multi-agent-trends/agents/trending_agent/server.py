from fastapi import FastAPI
from agents.trending_agent.agent import find_trends

app = FastAPI(title="Trending Topics Agent")


@app.get("/find_trends")
def find():
    return find_trends()
