# MediBlaze-AI-Medical-Assistant (RAG)
MediBlaze is an AI-powered medical chatbot built using FastAPI, Retrieval-Augmented Generation (RAG), Pinecone, and Azure OpenAI / GitHub Models.
It provides safe, informational medical guidance by retrieving relevant medical knowledge and generating AI responses with streaming support.
# Features
This project is an AI-powered medical assistant that helps identify possible diseases based on user-entered symptoms. It leverages real-time streaming responses and Retrieval-Augmented Generation (RAG) with Pinecone for accurate and context-aware answers. The system also supports tool-based agents, such as locating nearby hospitals, and includes content-filter–aware error handling through Azure Responsible AI.Docker and Docker Compose support make deployment easy, and sensitive information is securely managed via environment variables.
# Tech Stack
Backend: Python, FastAPI

AI / LLM: Azure OpenAI (GitHub Models – GPT-4)

Vector Database: Pinecone

RAG: Custom document ingestion & retrieval

Deployment: Docker, Docker Compose

Environment Management: venv, .env

Testing: JSON-based API test cases
# Setup Instructions (Local)
1.Clone the repository

2.Create & activate virtual environment

````bash
python -m venv venv
.\venv\Scripts\Activate
````

3.Install dependencies

````bash
pip install -r requirements.txt
````

4.Configure environment variables

PINECONE_API_KEY=your_key

GITHUB_TOKEN=your_token

5.Setup Pinecone index

````bash
python setup_pinecone.py
````

6.Upload documents for RAG

7.Run the application

````bash
python main.py
````
# Run with Docker
````bash
docker-compose up --build
````
# Output Image
<img width="1562" height="997" alt="image" src="https://github.com/user-attachments/assets/021811e1-07c0-46a4-aae8-b8d67b1df340" />

# Outcomes
1.Building a production-style AI backend

2.Implementing RAG with Pinecone

3.Handling LLM safety & content filters

4.Secure API key management

5.Dockerized deployment
