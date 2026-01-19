class AgentMemory:
    def __init__(self):
        self.state = {}

    def remember(self, key, value):
        self.state[key] = value

    def recall(self, key):
        return self.state.get(key)
