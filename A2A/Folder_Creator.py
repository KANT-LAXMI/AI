import os

PROJECT_NAME = "azure-multi-agent-trends"

# Folder & file structure definition
STRUCTURE = {
    PROJECT_NAME: {
        "README.md": "",
        ".env": "",
        "requirements.txt": "",

        "common": {
            "__init__.py": "",
            "azure_openai_client.py": "",
            "web_search.py": "",
        },

        "agents": {
            "__init__.py": "",

            "trending_agent": {
                "__init__.py": "",
                "agent.py": "",
                "server.py": "",
            },

            "analyzer_agent": {
                "__init__.py": "",
                "agent.py": "",
                "server.py": "",
            },

            "host_agent": {
                "__init__.py": "",
                "agent.py": "",
                "server.py": "",
            },
        },

        "run_all.py": "",
    }
}


def create_structure(base_path, structure):
    """
    Recursively create folders and files
    """
    for name, content in structure.items():
        path = os.path.join(base_path, name)

        # If content is dict → folder
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)

        # Else → file
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)


if __name__ == "__main__":
    create_structure(".", STRUCTURE)
    print(f"✅ Project structure '{PROJECT_NAME}' created successfully!")
