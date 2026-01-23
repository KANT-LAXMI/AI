from fastapi import FastAPI
from agents.host_agent.agent import run_full_analysis

app = FastAPI(title="Trend Analysis Host")


@app.get("/run")
def run():
    return run_full_analysis()
