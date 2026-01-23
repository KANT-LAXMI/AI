import subprocess

services = [
    ("Trending Agent", "agents/trending_agent/server.py", 10020),
    ("Analyzer Agent", "agents/analyzer_agent/server.py", 10021),
    ("Host Agent", "agents/host_agent/server.py", 10022),
]

for name, path, port in services:
    subprocess.Popen([
        "uvicorn",
        path.replace("/", ".").replace(".py", ":app"),
        "--port",
        str(port)
    ])

print("All agents running 🚀")
