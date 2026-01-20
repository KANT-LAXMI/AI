
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("SimpleServer")

@mcp.tool()
def hello_world(name: str = "World"):
    return {"message": f"Hello, {name}!"}

@mcp.tool()
def add(a: int, b: int):
    return a + b

if __name__ == "__main__":
    mcp.run("stdio")
