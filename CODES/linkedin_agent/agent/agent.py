from tools.pdl_search import search_people
from tools.export_excel import export_to_excel
from memory import AgentMemory

class PDLAgent:
    def __init__(self):
        self.memory = AgentMemory()

    def run(self, keyword):
        print("🧠 Agent deciding workflow...")

        if not keyword:
            raise ValueError("Keyword cannot be empty")

        print("🔍 Agent calling PDL Search tool...")
        
        total, people = search_people(keyword) 

        self.memory.remember("last_search_total", total)

        print("📁 Agent exporting results...")
        file = export_to_excel(keyword, total, people)

        self.memory.remember("last_file", file)

        return {
            "keyword": keyword,
            "total": total,
            "file": file
        }
