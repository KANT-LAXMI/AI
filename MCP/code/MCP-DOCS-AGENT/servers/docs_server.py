from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import httpx
import json
import os
from bs4 import BeautifulSoup

load_dotenv()

# ===============================
# MCP SERVER (Protocol Handler)
# ===============================
mcp = FastMCP("docs")

USER_AGENT = "docs-app/1.0"
SERPER_URL = "https://google.serper.dev/search"

docs_urls = {
    "langchain": "python.langchain.com/docs",
    "llama-index": "docs.llamaindex.ai/en/stable",
    "openai": "platform.openai.com/docs",
    "mcp": "modelcontextprotocol.io"
}

# ===============================
# SEARCH LAYER
# ===============================
async def search_web(query: str) -> dict:
    payload = json.dumps({"q": query, "num": 2})
    headers = {
        "X-API-KEY": os.getenv("SERPER_API_KEY"),
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                SERPER_URL, headers=headers, data=payload, timeout=30.0
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            return {"organic": []}

# ===============================
# FETCH + PARSE DOCS
# ===============================
async def fetch_url(url: str) -> str:
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}) as client:
        try:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            main = soup.find("main") or soup.find("article")
            text = main.get_text(separator="\n", strip=True) if main else soup.get_text()

            if len(text) > 8000:
                text = text[:8000] + "... [truncated]"

            return text
        except Exception as e:
            return f"Error fetching page: {e}"

# ===============================
# MCP TOOL
# ===============================
@mcp.tool()
async def get_docs(query: str, library: str) -> str:
    """
    Search official documentation and return live content.

    Args:
        query: What to search for
        library: langchain | openai | mcp | llama-index
    """
    if library not in docs_urls:
        return f"Unsupported library. Choose from: {', '.join(docs_urls.keys())}"

    search_query = f"site:{docs_urls[library]} {query}"
    results = await search_web(search_query)

    if not results["organic"]:
        return "No documentation found."

    combined = ""
    for i, item in enumerate(results["organic"]):
        combined += f"\n\nSOURCE: {item.get('title')}\nURL: {item.get('link')}\n\n"
        combined += await fetch_url(item["link"])

    return combined


# ===============================
# TRANSPORT LAYER
# ===============================
if __name__ == "__main__":
    mcp.run(transport="stdio")
