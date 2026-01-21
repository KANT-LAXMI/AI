"""
Healthcare Agent
Combines LangChain ReAct agent with MCP tools for intelligent medical assistance
Integrates with both medical and pharmacy MCP servers
"""

import asyncio
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from config.settings import OLLAMA_CONFIG, MCP_SERVERS
from typing import Optional, Dict, Any
import json

# ===============================
# HEALTHCARE AGENT CLASS
# ===============================
class HealthcareAgent:
    """
    Intelligent healthcare agent that uses LLM + MCP tools to assist with:
    - Patient information retrieval
    - Medical history analysis
    - Appointment scheduling
    - Prescription management
    - Drug interaction checking
    - Vital signs tracking
    """
    
    def __init__(self):
        """Initialize the healthcare agent with LLM and MCP clients"""
        self.model = None
        self.mcp_client = None
        self.agent = None
        self.tools = None
        self._initialized = False
    
    async def initialize(self):
        """
        Initialize the agent with LLM model and MCP tools.
        This must be called before using the agent.
        """
        if self._initialized:
            return
        
        # ===============================
        # MODEL LAYER (LLM)
        # ===============================
        self.model = ChatOllama(
            model=OLLAMA_CONFIG["model"],
            base_url=OLLAMA_CONFIG["base_url"],
            temperature=OLLAMA_CONFIG["temperature"]
        )
        
        # ===============================
        # MCP CLIENT (Protocol Handler)
        # ===============================
        # Connect to both medical and pharmacy MCP servers
        self.mcp_client = MultiServerMCPClient(MCP_SERVERS)
        
        # ===============================
        # LOAD MCP TOOLS
        # ===============================
        self.tools = await self.mcp_client.get_tools()
        
        print(f"✅ Loaded {len(self.tools)} tools from MCP servers")
        print(f"📋 Available tools:")
        for tool in self.tools:
            print(f"   - {tool.name}: {tool.description[:80]}...")
        
        # ===============================
        # AGENT LAYER (ReAct)
        # ===============================
        self.agent = create_react_agent(
            model=self.model,
            tools=self.tools
        )
        
        self._initialized = True
        print("✅ Healthcare Agent initialized successfully")
    
    async def query(self, question: str) -> Dict[str, Any]:
        """
        Query the healthcare agent with a question.
        
        Args:
            question: Natural language question about patients, appointments, etc.
        
        Returns:
            Dictionary containing the agent's response and metadata
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # ===============================
            # RUN AGENT
            # ===============================
            result = await self.agent.ainvoke(
                {"messages": question}
            )
            
            # Extract the final answer
            final_message = result["messages"][-1]
            
            # Parse tool calls if any
            tool_calls = []
            for msg in result["messages"]:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_calls.append({
                            "tool": tool_call.get("name", "unknown"),
                            "args": tool_call.get("args", {})
                        })
            
            return {
                "status": "success",
                "answer": final_message.content,
                "tool_calls": tool_calls,
                "full_conversation": [
                    {
                        "role": msg.type if hasattr(msg, 'type') else "unknown",
                        "content": msg.content if hasattr(msg, 'content') else str(msg)
                    }
                    for msg in result["messages"]
                ]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "answer": f"I encountered an error: {str(e)}"
            }
    
    async def close(self):
        """Clean up resources"""
        if self.mcp_client:
            # MCP client cleanup if needed
            pass
        self._initialized = False


# ===============================
# STANDALONE USAGE EXAMPLE
# ===============================
async def main():
    """
    Example of using the Healthcare Agent standalone.
    This demonstrates various queries the agent can handle.
    """
    agent = HealthcareAgent()
    await agent.initialize()
    
    # Example queries
    queries = [
        "What are the current system statistics?",
        "Search for patients named 'John'",
        "Get details for patient ID 1",
        "What medications is patient 1 currently taking?",
        "Check for drug interactions if we prescribe Aspirin to patient 2",
        "Show me upcoming appointments for this week",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}: {query}")
        print(f"{'='*80}")
        
        result = await agent.query(query)
        
        if result["status"] == "success":
            print(f"\n📋 Answer:")
            print(result["answer"])
            
            if result["tool_calls"]:
                print(f"\n🔧 Tools Used:")
                for tool_call in result["tool_calls"]:
                    print(f"   - {tool_call['tool']}")
        else:
            print(f"\n❌ Error: {result['error']}")
        
        # Small delay between queries
        await asyncio.sleep(1)
    
    await agent.close()


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())