import asyncio
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

# ===============================
# MODEL LAYER
# ===============================
model = ChatOllama(
    model="llama3.2:3b",
    temperature=0.0,
    max_new_tokens=500
)

# ===============================
# CLIENT APPLICATION
# ===============================
async def run_agent(question: str):

    # ❌ DO NOT use "async with"
    # ✅ Create client normally
    client = MultiServerMCPClient(
        {
            "docs": {
                "command": "python",
                "args": ["servers/docs_server.py"],
                "transport": "stdio",
            }
        }
    )

    # ===============================
    # LOAD MCP TOOLS
    # ===============================
    tools = await client.get_tools()

    # ===============================
    # AGENT LAYER (ReAct)
    # ===============================
    agent = create_react_agent(
        model=model,
        tools=tools
    )

    # ===============================
    # RUN AGENT
    # ===============================
    result = await agent.ainvoke(
        {"messages": question}
    )

    print("\n=== FINAL ANSWER ===\n")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    question = "Explain MCP transport stdio vs sse"
    asyncio.run(run_agent(question))
    
# run python agents/docs_agent.py
# === FINAL ANSWER ===

# The `stdio` and `sse` options in the MCP (Microsoft Component Platform) transport are used to configure how data is sent and received over a network connection.
# **stdio**
# `stdio` stands for "standard input/output". When using `stdio`, the MCP transport uses standard TCP/IP sockets to send and receive data. This means that the data is sent as a stream of bytes, with each byte being sent individually. The receiving end can then read this stream of bytes and reconstruct the original data.
# **sse**
# `sse` stands for "streaming socket endpoint". When using `sse`, the MCP transport uses a specialized TCP/IP socket that allows for bidirectional streaming of data between two endpoints. This means that both the sender and receiver can send data to each other in real-time, without having to wait for the entire stream to be sent.
# The key differences between `stdio` and `sse` are:
# *   **Streaming**: `sse` provides a more efficient way to transfer large amounts of data, as it allows for bidirectional streaming. `stdio`, on the other hand, sends data in a linear fashion.
# *   **Latency**: `sse` typically has lower latency than `stdio`, since data can be sent and received simultaneously.
