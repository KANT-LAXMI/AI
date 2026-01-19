from agent import PDLAgent

if __name__ == "__main__":
    agent = PDLAgent()

    keyword = input("Enter keyword: ").strip()

    result = agent.run(keyword)

    print("\n✅ Agent Completed")
    print(f"Keyword: {result['keyword']}")
    print(f"Estimated Professionals: {result['total']}")
    print(f"Excel File: {result['file']}")
