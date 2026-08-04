# 🤖 LangGraph ChatBot & LLM Workflows

> **A modular collection of advanced LangGraph-powered LLM workflows and interactive AI chat applications built using Python, LangGraph, LangChain, and Streamlit.**

---

# 📑 Table of Contents

- [📌 Project Overview](#-project-overview)
- [🎯 Problem Statement](#-problem-statement)
- [📂 Dataset](#-dataset)
- [🛠️ Tools & Technologies](#️-tools--technologies)
- [🏗️ Project Architecture](#️-project-architecture)
- [⚙️ Workflow Implementation](#️-workflow-implementation)
- [🧠 Key Features](#-key-features)
- [📊 Project Structure](#-project-structure)
- [🚀 How to Run the Project](#-how-to-run-the-project)
- [📈 Results & Conclusion](#-results--conclusion)
- [🔮 Future Improvements](#-future-improvements)
- [👨‍💻 Author & Contact](#-author--contact)

---

# 📌 Project Overview

This project serves as a comprehensive development environment for building, experimenting with, and deploying **LangGraph-based Large Language Model (LLM) applications**.

It demonstrates how modern **stateful AI workflows** can be developed by combining **LangGraph**, **LangChain**, and **Streamlit** into reusable, modular components. The repository includes standalone workflow examples, multi-node graph implementations, and production-style chatbot interfaces featuring **real-time response streaming** and **multi-threaded execution**.

### 🎯 Project Objectives

- Build modular LLM workflows using LangGraph.
- Demonstrate graph-based orchestration instead of linear pipelines.
- Develop interactive chatbot interfaces using Streamlit.
- Showcase different conversational architectures and agent workflows.
- Provide reusable templates for future AI applications.

---

# 🎯 Problem Statement

Traditional LLM applications generally follow **linear execution pipelines**, making it difficult to handle:

- Complex decision-making
- Conditional execution
- Multi-agent collaboration
- Looping workflows
- Stateful conversations
- Error correction and retries

This project explores how **LangGraph's graph-based architecture** enables deterministic, scalable, and maintainable AI workflows by introducing state management and modular execution paths.

---

# 📂 Dataset

This project currently **does not rely on a static dataset**.

Instead, it operates primarily on:

- User prompts
- Large Language Models
- Dynamic prompt engineering
- Stateful graph execution

> **Note:** Future versions may integrate vector databases, RAG pipelines, or external knowledge bases for enhanced retrieval capabilities.

---

# 🛠️ Tools & Technologies

### Programming Language

- Python 3.10+

### LLM Frameworks

- LangGraph
- LangChain

### Frontend

- Streamlit

### Environment & Configuration

- Python Virtual Environment (venv)
- dotenv

### Version Control

- Git
- GitHub

### APIs

- OpenAI API
- Anthropic API

---

# 🏗️ Project Architecture

The project is divided into three independent modules:

### 🤖 ChatBot

Contains the basic chatbot implementation for conversational interactions.

### 💬 ChatBot with UI

Implements interactive Streamlit-based chat interfaces supporting:

- Live response streaming
- Multi-threaded execution
- LangGraph backend integration

### 🧠 LangGraph Model

A collection of standalone workflow examples demonstrating different graph-based execution patterns including:

- Conditional execution
- Prompt chaining
- Mathematical reasoning
- Sentiment analysis
- Creative content generation

---

# ⚙️ Workflow Implementation

The repository demonstrates multiple LangGraph workflow patterns.

### 🔹 State Graph Architectures

- Multi-node execution
- Conditional routing
- Stateful transitions
- Graph-based orchestration

### 🔹 Sequential Prompt Chaining

- Context passing
- Multi-step reasoning
- Output refinement
- Chained LLM execution

### 🔹 Interactive Streaming

- Token streaming
- Real-time responses
- Improved user experience

### 🔹 Multi-threaded Processing

- Parallel execution
- Thread-safe Streamlit interface
- Responsive UI handling

---

# 🧠 Key Features

✅ Modular LangGraph workflows

✅ Interactive Streamlit chatbot

✅ Real-time response streaming

✅ Multi-threaded chat execution

✅ Multiple workflow examples

✅ Graph-based state management

✅ Prompt chaining demonstrations

✅ Easy project extensibility

✅ Production-ready project structure

---

# 📊 Project Structure

```text
LangGraph-ChatBot/
│
├── ChatBot/
│   └── basic_chatbot.py
│
├── ChatBot_with_UI/
│   ├── langgraph_backend.py
│   ├── streamlit_frontend.py
│   ├── streamlit_frontend_streaming.py
│   └── streamlit_frontend_threading.py
│
├── LangGraph_Model/
│   ├── 1_bmi_calculator.py
│   ├── 2_simple_llm_wflow.py
│   ├── 3_prompt_chaining.py
│   ├── 4_batsman_wflow.py
│   ├── 5_quadratic_eq.py
│   ├── 6_sentiment_review.py
│   └── 7_insta_post.py
│
├── requirements.txt
├── .env
└── README.md
```

---

# 🚀 How to Run the Project

## 1️⃣ Clone the Repository

```bash
git clone <repository-url>
cd LangGraph-ChatBot
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv myenv
```

### Windows

```bash
myenv\Scripts\activate
```

### Linux / macOS

```bash
source myenv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Configure Environment Variables

Create a `.env` file inside the project root.

```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

---

## 5️⃣ Run Standalone Workflow

Example:

```bash
python LangGraph_Model/1_bmi_calculator.py
```

You can similarly execute any workflow inside the **LangGraph_Model** directory.

---

## 6️⃣ Launch the Chatbot UI

```bash
streamlit run ChatBot_with_UI/streamlit_frontend_streaming.py
or
streamlit run ChatBot_with_UI/streamlit_frontend_streaming.py --server.fileWatcherType none
```

Alternatively,

```bash
streamlit run ChatBot_with_UI/streamlit_frontend.py
```

---

# 📈 Results & Conclusion

The project successfully demonstrates how **LangGraph** can orchestrate complex LLM workflows beyond traditional sequential pipelines.

### Key Outcomes

- Stable graph-based execution
- Modular architecture for scalability
- Separation of frontend and backend logic
- Efficient state management
- Real-time response streaming
- Improved user interaction through asynchronous execution

The repository serves as an excellent foundation for building advanced AI assistants, autonomous agents, and production-ready conversational systems.

---

# 🔮 Future Improvements

- [ ] Integrate Retrieval-Augmented Generation (RAG)
- [ ] Add long-term conversational memory
- [ ] Support vector databases (FAISS, Pinecone, ChromaDB)
- [ ] Implement external tool calling
- [ ] Multi-agent collaboration workflows
- [ ] Deploy using Docker
- [ ] Cloud deployment (AWS/Azure/GCP)
- [ ] User authentication and session management
- [ ] Conversation history persistence
- [ ] Monitoring and logging dashboards

---

# 👨‍💻 Author & Contact

**Author:** Akash Singh

📧 Email: akash0112002@gmail.com

💼 LinkedIn:

💻 GitHub:

🌐 Portfolio:

---

## ⭐ If you found this project useful, consider giving it a Star on GitHub!