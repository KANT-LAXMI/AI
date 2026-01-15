# SEMANTIC KERNEL

Semantic Kernel is a lightweight AI orchestration framework by Microsoft that enables developers to build intelligent, agent-based applications by combining large language models, memory, prompts, and native code in a structured and secure way.
The Semantic Kernel is particularly useful for building AI agents and automating business processes by combining prompts with existing APIs to perform actions.

# Kernel in Semantic Kernel

The **Kernel** is the central component of the **Semantic Kernel** framework. At its core, the kernel acts as a **Dependency Injection (DI) container** that manages all the services, plugins, and configurations required to run an AI application.

![alt text](image-1.png)

By registering all AI services (such as Large Language Models), native code plugins, and supporting services with the kernel, developers enable seamless orchestration where the AI can automatically access and use these components whenever needed. Conceptually, the kernel functions as an **abstract execution engine** that drives the complete flow of the application.

Because the kernel contains all required AI and native services, it is used by nearly every component within the Semantic Kernel SDK. Any prompt execution, plugin invocation, or agent workflow relies on the kernel to retrieve the appropriate services and execute logic. As a result, the kernel is always available whenever prompts or code are run within Semantic Kernel.

## Prompt Execution Lifecycle

When a prompt is invoked through the kernel, it orchestrates the entire lifecycle:

- Selects the most appropriate AI service
- Builds the final prompt using the defined prompt templates
- Sends the prompt to the AI service
- Receives and parses the AI response
- Returns the processed output back to the application

## Middleware and Observability

Throughout this lifecycle, the kernel provides **middleware and event hooks** that allow developers to intercept and extend behavior at each stage. This enables:

- Centralized logging and monitoring
- User-facing status updates
- Enforcement of **Responsible AI** practices

All of these capabilities are managed from a **single, unified control point**.

`Before building a kernel, you should first understand the two types of components that exist:`

## Main Components of Semantic Kernel

## `1) AI Service Connectors` -> Facilitate integration with various AI services.

### 🔹 What are Semantic Kernel AI service connectors?

AI service connectors are like **adapters**.  
They allow Semantic Kernel to communicate with different AI providers (such as Azure OpenAI, OpenAI, etc.) using a **single common interface**.

👉 This means:

- You write code **once**
- You can switch AI providers **without changing much code**

---

### 🔹 What AI services do they support?

Semantic Kernel supports multiple types of AI services, including:

- 💬 **Chat Completion** → chatbots, assistants
- ✍️ **Text Generation** → summaries, explanations, content creation
- 🧠 **Embedding Generation** → search, similarity, RAG
- 🖼️ **Text to Image** → generate images from text
- 👁️ **Image to Text** → OCR, image understanding
- 🔊 **Text to Audio** → text-to-speech
- 🎧 **Audio to Text** → speech-to-text

All of these services are available through the **same interface**.

In the table below, we can see the services that are supported by each of the SDKs.

![alt text](image-6.png)

---

### 🔹 What happens when you register an AI service with the Kernel?

When an AI service is registered with the Kernel:

✅ **Chat Completion** or **Text Generation**

- Are used **automatically** by the Kernel
- Whenever you run a prompt or request text generation

🚫 **Other services** (images, audio, embeddings, etc.):

- Are **not used automatically**
- Must be **explicitly called in code**

---

### 🔹 Simple example

Think of the Kernel as a **manager** 👔:

- If you say: _"Generate some text"_  
  👉 The Kernel automatically uses **Chat or Text Generation**

- If you say: _"Create an image"_ or _"Generate embeddings"_  
  👉 You must **explicitly specify** which AI service to use

---

### 🔹 One-line summary

> Semantic Kernel connects to multiple AI services using a single interface, but by default it automatically uses only chat or text generation—other services must be explicitly invoked.

## `2) Vector Store Connectors` -> Provide interfaces to connect with vector databases for efficient storage and retrieval of embeddings.

Semantic Kernel Vector Store connectors provide an abstraction layer that exposes vector stores from different providers through a **common interface**. This allows developers to integrate multiple vector database technologies without changing application logic.

By default, the Kernel does **not automatically use any registered vector store**, since vector stores are primarily designed for retrieval and search operations rather than direct text generation.

However, **Vector Search can be exposed as a plugin** to the Kernel. When exposed as a plugin:

- Vector search becomes available to **prompt templates**
- Chat Completion AI models can retrieve relevant data before generating responses
- Enables powerful **Retrieval-Augmented Generation (RAG)** workflows

---

### 💡 Tip

> To enable semantic search or RAG scenarios in Semantic Kernel, expose Vector Search as a plugin so it can be accessed by prompt templates and Chat Completion models.

## `3) Functions and Plugins` -> Allow extension of the kernel’s functionality by incorporating native code and AI services as plugins. (Function Calls)

![alt text](image-3.png)

In Semantic Kernel, **plugins** are named containers that group related **functions**.  
Each plugin can contain one or more functions.

When plugins are registered with the Kernel, the Kernel can use them in two main ways:

1. **Expose functions to the Chat Completion AI**

   - Allows the AI model to see available functions
   - Enables the AI to choose and invoke functions when needed

2. **Make functions available to prompt templates**
   - Functions can be called during prompt template rendering
   - Useful for dynamic data, calculations, or retrieval

---

## 🔹 Sources of Functions

Functions in Semantic Kernel can be created from various sources, including:

- Native code (C#, Python, Java)
- OpenAPI specifications (external REST APIs)
- `ITextSearch` implementations for RAG scenarios
- Prompt templates that generate AI-powered responses

---

This flexible function and plugin model enables seamless collaboration between **AI models and native code**, supporting intelligent and agent-based application workflows.

## `4) Prompt Templates` -> Enable creation of reusable prompts with variable interpolation and function execution for consistent AI interactions.

Prompt templates allow developers or prompt engineers to define templates that combine:

- Context and instructions for the AI
- User input placeholders
- Outputs from functions or plugins

A prompt template may contain instructions for the **Chat Completion AI model**, placeholders for user input, and **hardcoded calls to plugins** that must be executed before invoking the Chat Completion model.

---

## 🔹 Ways to Use Prompt Templates

Prompt templates can be used in two ways:

1. **As the starting point of a Chat Completion flow**

   - The Kernel renders the template
   - Executes any hardcoded function references
   - Invokes the Chat Completion AI model using the rendered prompt

2. **As a plugin function**
   - The template is registered as a plugin function
   - It can be invoked like any other function
   - It may be selected automatically by the Chat Completion AI model

---

## 🔹 Prompt Template Execution

When a prompt template is used:

1. The template is rendered
2. Any hardcoded function references are executed
3. The rendered prompt is passed to the Chat Completion AI model
4. The AI generates a response
5. The result is returned to the caller

If the prompt template is registered as a plugin function, it may be invoked by the AI model itself. In this case, the caller is **Semantic Kernel**, acting on behalf of the AI model.

---

## 🔹 Complex Execution Flow Example

Consider the following scenario:

- Prompt Template **A** is registered as a plugin function
- Prompt Template **B** is used to start the Chat Completion flow

If **B** contains a hardcoded call to **A**, the execution flow is:

1. Rendering of **B** begins and a reference to **A** is detected
2. **A** is rendered
3. The rendered output of **A** is passed to the Chat Completion AI model
4. The AI response is returned to **B**
5. Rendering of **B** completes
6. The rendered output of **B** is passed to the Chat Completion AI model
7. The final AI response is returned to the caller

---

## 🔹 AI-Initiated Function Invocation

Even if **B** does not explicitly reference **A**, the Chat Completion AI model may still decide to invoke **A** when:

- Function calling is enabled
- The AI determines that **A** provides required data or functionality

---

## 🔹 Why Register Prompt Templates as Plugin Functions?

Registering prompt templates as plugin functions allows functionality to be defined using **human language instead of code**. By separating this logic into plugins:

- The AI can reason about each task independently
- Complex workflows become modular and reusable
- The AI can focus on one problem at a time
- Overall success rates of AI-driven workflows improve

![alt text](image-4.png)

## `5) Filters` -> Provide mechanisms to validate permissions and control the execution flow within the kernel.

Filters provide a way to take custom action before and after specific events during the chat completion flow. These events include:

- Before and after function invocation.
- Before and after prompt rendering.
- Filters need to be registered with the kernel to get invoked during the chat completion flow.

Note that since prompt templates are always converted to KernelFunctions before execution, both function and prompt filters will be invoked for a prompt template. Since filters are nested when more than one is available, function filters are the outer filters and prompt filters are the inner filters.

![alt text](image-5.png)

# Coding an Orchestrator

Let’s proceed to build an agent orchestrator. Specifically, we will create a simple autonomous AI-driven pizza ordering system.

The problem we are going to develop here aims to reduce the manual effort involved in taking user orders by automating the process with AI autonomously without compromising security and order miss in a real-world scenario.

Here is the architecture:

![alt text](image-7.png)

We have a Pizza Orchestrator that is equipped with the following functionalities:

- get_pizza_wallet_balance
- get_available_pizza
- order_pizza.

It can inform users about the available pizzas, place an order based on the user’s choice, and check the user’s wallet balance similar to how Amazon shopping wallets work.

## Package Installations and setup

```
pip install semantic-kernel
```

## Import the followings

```python
# asyncio is used because Semantic Kernel chat calls are async (non-blocking)
import asyncio

# -----------------------------
# 1. Kernel (Main Orchestrator)
# -----------------------------
# Kernel is the heart of Semantic Kernel
# It connects AI models + plugins (functions) + orchestration logic
from semantic_kernel import Kernel

# -----------------------------
# 2. Chat History (Short-term memory)
# -----------------------------
# ChatHistory stores the full conversation (system, user, assistant messages)
from semantic_kernel.contents import ChatHistory

# -----------------------------
# 3. Kernel Function Decorator
# -----------------------------
# kernel_function is a decorator
# It tells Semantic Kernel: "This Python function can be called by the LLM"
from semantic_kernel.functions import kernel_function

# -----------------------------
# 4. Azure OpenAI Chat Service
# -----------------------------
# AzureChatCompletion = Azure OpenAI chat model connector
# AzureChatPromptExecutionSettings = controls model behavior (function calling, temperature, etc.)
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureChatPromptExecutionSettings,
)

# -----------------------------
# 5. Function Calling Behavior
# -----------------------------
# FunctionChoiceBehavior.Auto means:
# LLM automatically decides when to call a function
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
```

## Defining the Plugin

A single Plugin can contain number of functions

```python
# =====================================================
# PLUGIN: Pizza Business Logic
# =====================================================
# Plugin = a group of functions that the AI agent can use
class PizzaPlugin:

    # @kernel_function exposes this method to the LLM
    # The LLM can "see" this function and call it when needed
    @kernel_function(
        description="Checks balance amount in rupees on users pizza wallet; returns the balance amount"
    )
    def get_pizza_wallet_balance(self, wallet_password: str):
        # This print is just for us (developer) to know function was called
        print("Invoked get_pizza_wallet_balance function !!")

        # Static wallet balance (mock data)
        balance = 144.34

        # Whatever we return becomes the function response sent to the LLM
        return f"balance : Rs.{balance}"

    # This function tells available pizzas
    @kernel_function(description="Checks for available pizzas and return them.")
    def get_available_pizza(self):

        # Static pizza menu data
        pizzas = {
            "Pizza 1": {"Name": "Bryon's Bigdamaka pizza", "Price": 180.76},
            "Pizza 2": {"Name": "Gramin's Small Pizza", "Price": 129.87},
            "Pizza 3": {"Name": "Jaorin's Special Pizza", "Price": 239.76},
        }

        print("Invoked get_available_pizza function !!")

        # Return must be string / JSON-like so LLM can understand it
        return str(pizzas)

    # Function to place pizza order
    @kernel_function(
        description="Order a pizza with the given pizza name and user wallet balance; return confirmation message"
    )
    def order_pizza(self, pizza_name: str, pizza_price: float, wallet_balance: float):
        print("Invoked Pizza order function !!")

        # Business logic: check wallet balance
        if wallet_balance < pizza_price:
            return (
                f"Your wallet balance is insufficient to place an order for {pizza_name}."
            )

        return f"Your order for {pizza_name} has been placed successfully."

```

Here, the functions decorated with the `@kernel_function` decorator are automatically converted into JSON schema and sent to the model, something we often have to define manually in other frameworks.

But yeah, we may lose some control from a developer perspective when things become more abstract.

## The Main Function

```python
# ---------------- Main Program ----------------
async def main():

    # Create Kernel instance
    # Kernel manages:
    # - AI model
    # - Plugins
    # - Function calling
    kernel = Kernel()

    # Register PizzaPlugin inside the kernel
    # plugin_name is how LLM identifies this plugin
    kernel.add_plugin(PizzaPlugin(), plugin_name="OrderPizzaPlugin")

    # Create Azure OpenAI chat service
    chat_service = AzureChatCompletion(
        deployment_name="gpt-4o-mini",   # Azure OpenAI model deployment name
        endpoint="https://<your-resource-name>.openai.azure.com/",  # Azure endpoint
        api_key="YOUR_AZURE_OPENAI_API_KEY",  # Azure OpenAI key
    )

    # Attach chat service to kernel
    kernel.add_service(chat_service)

    # ChatHistory stores entire conversation context
    chat_history = ChatHistory()

    # System message defines agent personality and role
    chat_history.add_system_message(
        "Your name is 'Pizzer'. You are a pizza ordering agent. "
        "You can check wallet balance, list pizzas, and place orders."
    )

    # Execution settings control model behavior
    execution_settings = AzureChatPromptExecutionSettings()

    # Auto = model decides when to call plugin functions
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    # Infinite chat loop
    while True:
        # Take user input from console
        user_input = input("Enter your message >>> ")

        # Exit condition
        if user_input.lower() == "q":
            print("Exiting...")
            break

        # Add user message to conversation history
        chat_history.add_user_message(user_input)

        # Ask Azure OpenAI for response
        # Kernel is passed so model can call functions
        response = await chat_service.get_chat_message_content(
            chat_history=chat_history,
            settings=execution_settings,
            kernel=kernel,
        )

        # Convert response object to string
        response_text = str(response)

        # Store assistant response in chat history
        chat_history.add_assistant_message(response_text)

        # Print response
        print("Response from agent >>>", response_text)


# Python entry point
if __name__ == "__main__":
    # Run async main function
    asyncio.run(main())
```

![alt text](image-8.png)

## Observation

I ran the code and asked for the Bryon’s Bigdamaka Pizza. It prompted me to enter my wallet password, which I provided. After that, it informed me that I didn’t have sufficient balance in my pizza wallet. The price of the pizza was 180.76. Then, I asked for Gramin’s Small Pizza, which costs 129.76 an amount less than my available balance. Finally it placed an order!!
