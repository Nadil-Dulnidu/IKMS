# 🧠 IKMS - Intelligent Knowledge Management System

IKMS (Intelligent Knowledge Management System) is a modern, AI-powered knowledge management platform that enables users to upload, organize, and intelligently query their documents. Built with a multi-agent architecture, IKMS leverages advanced AI techniques including LangGraph orchestration, vector embeddings, and retrieval-augmented generation (RAG) to provide accurate, context-aware answers from your personal knowledge base.

## 🤔 Problem Space

### Problems to Solve / Requirements to Create

IKMS addresses the growing challenge of managing and retrieving information from large document collections by providing:

* An AI-powered document query system with natural language understanding
* Secure, user-specific document storage and retrieval
* Multi-agent processing for accurate and verified answers
* Real-time streaming responses for better user experience

### 👉 Problem: Information Overload & Inefficient Document Search

#### Problem Statement:
Traditional document management systems rely on keyword-based search, which often fails to understand context and user intent. Users waste time manually searching through multiple documents to find relevant information.

#### Current Solution:
IKMS implements a multi-agent RAG system that:
- Analyzes user queries to understand intent
- Retrieves relevant document chunks using semantic search
- Critically evaluates context quality
- Generates comprehensive, verified answers

#### How do we know it is a problem?

* User Feedback: Users struggle to find specific information across multiple PDF documents
* Metrics: Traditional search returns irrelevant results when queries are complex or contextual
* Evidence: Research shows semantic search outperforms keyword search by 40-60% in accuracy

### 👉 Problem: Document Security & User Isolation

#### Problem Statement:
Multi-tenant document management systems must ensure complete data isolation between users while maintaining fast query performance.

#### Current Solution:
IKMS uses Pinecone's namespace feature to create isolated vector stores for each user, ensuring:
- Complete data separation between users
- Secure authentication via Clerk JWT tokens
- User-specific retrieval that never accesses other users' documents

#### How do we know it is a problem?

* Security Requirements: Data privacy regulations (GDPR, CCPA) mandate strict user data isolation
* User Trust: Users need confidence that their documents are private and secure
* Evidence: Namespace-based isolation provides database-level security without performance overhead

### 👉 Problem: Answer Quality & Hallucination Prevention

#### Problem Statement:
AI language models can generate plausible-sounding but incorrect answers (hallucinations), especially when working with specialized or technical documents.

#### Current Solution:
IKMS implements a multi-agent verification pipeline:
1. **Query Planning Agent**: Decomposes complex queries into sub-questions
2. **Retrieval Agent**: Gathers relevant context from vector store
3. **Context Critic Agent**: Filters and ranks retrieved information
4. **Summarization Agent**: Generates draft answers from verified context
5. **Verification Agent**: Cross-checks answers against source material

#### How do we know it is a problem?

* Research: Studies show LLMs hallucinate 15-20% of the time without proper grounding
* User Reports: Users need citations and verifiable answers for critical information
* Evidence: Multi-agent verification reduces hallucination rates by 70-80%

### ⭐ Why Solve These Problems?

*Reason 1*: Organizations and individuals need efficient ways to manage and query growing document collections without manual effort.

*Reason 2*: AI-powered knowledge management improves productivity, reduces information retrieval time, and enables better decision-making based on accurate, verified information.

## 🎯 Project Goals

#### Project Objective

Build a scalable, secure, and intelligent knowledge management system that enables users to upload documents and receive accurate, AI-powered answers to their questions.

#### Project Goals

* Develop a web-based platform with secure user authentication (Clerk)
* Implement PDF document upload and intelligent indexing
* Create a multi-agent RAG system for accurate question answering
* Provide real-time streaming responses for better UX
* Ensure complete user data isolation using namespace-based vector storage
* Build an intuitive chat interface for natural language queries

## 👥 User Stories

### User Type: Knowledge Worker / Student

**Goals:**
* Upload and organize personal documents (research papers, notes, reports)
* Query documents using natural language
* Receive accurate, cited answers quickly

**Needs:**
* Simple document upload interface
* Fast, accurate search and retrieval
* Confidence in answer accuracy

**Characteristics:**
* Manages multiple documents across various topics
* Needs quick access to specific information
* Values accuracy and source attribution

## 🌟 Design Space

### UI Design

IKMS features a clean, modern chat interface built with Next.js and shadcn/ui components:

* **Dark Mode Support**: Elegant dark theme for reduced eye strain
* **Real-time Streaming**: Answers stream in real-time for immediate feedback
* **File Upload**: Drag-and-drop PDF upload with visual feedback
* **Responsive Design**: Works seamlessly across desktop and mobile devices
* **Tool Visualization**: Shows AI agent tool usage during processing

#### User Flow

**Document Upload & Query Flow**
1. User signs in via Clerk authentication
2. Upload PDF documents through chat interface
3. System indexes documents into user's private namespace
4. User asks questions in natural language
5. Multi-agent system processes query
6. Streaming response with verified answer
7. User can ask follow-up questions

### Technology Stack

#### Frontend - Next.js 16 with TypeScript

**Why Next.js?**
* **Server-Side Rendering**: Fast initial page loads and SEO benefits
* **React 19**: Latest React features with improved performance
* **TypeScript**: Type safety reduces bugs and improves developer experience
* **App Router**: Modern routing with layouts and streaming support

**Why Vercel AI SDK?**
* **Streaming Support**: Real-time response streaming for better UX
* **Tool Integration**: Seamless integration with LangChain tools
* **Type Safety**: Full TypeScript support for AI interactions

#### Backend - FastAPI with Python

**Why FastAPI?**
* **High Performance**: Async support for handling concurrent requests
* **Type Safety**: Pydantic models ensure data validation
* **OpenAPI Integration**: Automatic API documentation
* **Modern Python**: Leverages Python 3.12+ features

**Why LangGraph?**
* **Multi-Agent Orchestration**: Manages complex agent workflows
* **State Management**: Handles conversation state and checkpointing
* **Tool Integration**: Seamless integration with LangChain tools
* **Debugging**: Built-in visualization and debugging tools

#### Vector Database - Pinecone

**Why Pinecone?**
* **Managed Service**: No infrastructure management required
* **Namespace Support**: Built-in user isolation
* **High Performance**: Sub-100ms query latency
* **Scalability**: Handles millions of vectors effortlessly

#### Authentication - Clerk

**Why Clerk?**
* **Modern Auth**: Passwordless, social login, and MFA support
* **JWT Tokens**: Secure, stateless authentication
* **User Management**: Built-in user profiles and session handling
* **Developer Experience**: Easy integration with Next.js and FastAPI

#### LLM Provider - OpenAI

**Why OpenAI?**
* **GPT-4o**: State-of-the-art language understanding
* **Embeddings**: High-quality text embeddings for semantic search
* **Reliability**: Production-ready with 99.9% uptime
* **API Quality**: Well-documented, easy-to-use API

## 🏗️ Development Phase

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Client Layer                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Next.js 16 Frontend (TypeScript + shadcn/ui)          │ │
│  │  - Chat Interface                                       │ │
│  │  - File Upload                                          │ │
│  │  - Clerk Authentication                                 │ │
│  │  - Vercel AI SDK Integration                            │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTPS / JWT Auth
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Gateway Layer                       │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  FastAPI Backend (Python 3.12+)                        │ │
│  │  - REST API Endpoints                                   │ │
│  │  - JWT Token Verification                               │ │
│  │  - Request Validation                                   │ │
│  │  - Streaming Response Handler                           │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Multi-Agent Processing Layer               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  LangGraph Orchestration                               │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │  1. Query Planning Agent                         │  │ │
│  │  │     - Analyzes user query                        │  │ │
│  │  │     - Creates search strategy                    │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │  2. Retrieval Agent                              │  │ │
│  │  │     - Executes semantic search                   │  │ │
│  │  │     - Retrieves relevant chunks                  │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │  3. Context Critic Agent                         │  │ │
│  │  │     - Evaluates chunk relevance                  │  │ │
│  │  │     - Filters low-quality context                │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │  4. Summarization Agent                          │  │ │
│  │  │     - Generates draft answer                     │  │ │
│  │  │     - Cites source material                      │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │  5. Verification Agent                           │  │ │
│  │  │     - Cross-checks facts                         │  │ │
│  │  │     - Corrects inaccuracies                      │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Storage Layer                      │
│  ┌──────────────────────┐    ┌──────────────────────────┐  │
│  │  Pinecone Vector DB  │    │  PostgreSQL (Optional)   │  │
│  │  - User Namespaces   │    │  - Conversation History  │  │
│  │  - Document Chunks   │    │  - Checkpointing         │  │
│  │  - Embeddings        │    │  - State Management      │  │
│  └──────────────────────┘    └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    External Services Layer                   │
│  ┌──────────────────────┐    ┌──────────────────────────┐  │
│  │  OpenAI API          │    │  Clerk Authentication    │  │
│  │  - GPT-4o            │    │  - User Management       │  │
│  │  - Embeddings        │    │  - JWT Tokens            │  │
│  │  - Reasoning Models  │    │  - Session Handling      │  │
│  └──────────────────────┘    └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Development Workflow

#### Frontend (Next.js + TypeScript)

The frontend is built using Next.js 16 with TypeScript and shadcn/ui components:

* **Next.js App Router**: Modern routing with server components and streaming
* **TypeScript**: Full type safety across the application
* **Vercel AI SDK**: Handles streaming responses and tool visualization
* **shadcn/ui**: Accessible, customizable UI components
* **Clerk Integration**: Seamless authentication with JWT token management
* **Tailwind CSS**: Utility-first styling with dark mode support

The frontend communicates with the backend via RESTful APIs, sending user queries and receiving streaming responses in real-time.

#### Backend (FastAPI + LangGraph)

The backend is powered by FastAPI and LangGraph, managing all business logic and AI orchestration:

* **FastAPI Framework**: High-performance async API with automatic validation
* **JWT Authentication**: Verifies Clerk tokens and extracts user identity
* **LangGraph Orchestration**: Manages multi-agent workflow execution
* **Streaming Service**: Converts LangGraph events to Vercel AI SDK format
* **File Processing**: Handles PDF upload, parsing, and indexing
* **Logging**: Structured logging for debugging and monitoring

The backend implements a linear multi-agent pipeline that processes each query through five specialized agents.

#### Vector Database (Pinecone)

Pinecone stores document embeddings with user-specific namespaces:

* **Namespace Isolation**: Each user has a dedicated namespace for complete data separation
* **Semantic Search**: Retrieves relevant chunks based on embedding similarity
* **Metadata Storage**: Stores document metadata (filename, page number, etc.)
* **Fast Retrieval**: Sub-100ms query latency for real-time responses

Document chunks are embedded using OpenAI's `text-embedding-3-small` model and stored with metadata for citation.

#### Document Processing Pipeline

1. **Upload**: User uploads PDF through chat interface
2. **Validation**: Backend validates file type and size
3. **Parsing**: PyMuPDF4LLM extracts text and structure
4. **Chunking**: RecursiveCharacterTextSplitter creates 500-character chunks
5. **Embedding**: OpenAI generates vector embeddings
6. **Indexing**: Chunks stored in user's Pinecone namespace

### Multi-Agent Query Processing Flow

#### 1. Query Planning Agent
- **Purpose**: Analyzes the user's question and creates a search strategy
- **Input**: User query
- **Output**: Structured query plan with sub-questions
- **Model**: GPT-4o

#### 2. Retrieval Agent
- **Purpose**: Executes semantic search against the user's document collection
- **Input**: Query plan
- **Output**: Relevant document chunks
- **Tools**: Pinecone retrieval tool (user-specific namespace)
- **Model**: GPT-4o

#### 3. Context Critic Agent
- **Purpose**: Evaluates and filters retrieved chunks for relevance
- **Input**: Retrieved chunks + original query
- **Output**: Ranked, filtered chunks
- **Model**: GPT-4o

#### 4. Summarization Agent
- **Purpose**: Generates a comprehensive answer from verified context
- **Input**: Filtered chunks + query
- **Output**: Draft answer with citations
- **Model**: GPT-4o

#### 5. Verification Agent
- **Purpose**: Cross-checks the answer against source material
- **Input**: Draft answer + source chunks
- **Output**: Verified, corrected final answer
- **Model**: GPT-4o with reasoning capabilities

### Key Features of the Software

#### Multi-Agent RAG System
* Five specialized agents working in sequence
* Query decomposition for complex questions
* Context quality evaluation and filtering
* Answer verification and fact-checking

#### User-Specific Document Storage
* Namespace-based isolation in Pinecone
* Secure JWT authentication via Clerk
* Complete data privacy between users
* Fast, user-scoped retrieval

#### Real-Time Streaming Responses
* Vercel AI SDK integration
* Server-sent events (SSE) for streaming
* Tool execution visualization
* Progressive answer rendering

#### PDF Document Processing
* PyMuPDF4LLM for high-quality text extraction
* Recursive chunking with overlap
* Metadata preservation (page numbers, filenames)
* Automatic embedding generation

#### Conversation State Management
* LangGraph checkpointing (optional PostgreSQL)
* Thread-based conversation history
* State persistence across sessions

## 🚧 Challenges Faced & Solutions

### Problem 1: Streaming LangGraph Events to Vercel AI SDK

**Challenge**: LangGraph uses a different event format than Vercel AI SDK expects.

**Solution**: Built a custom streaming service that:
- Converts LangGraph events to Vercel AI SDK format
- Handles tool calls and outputs
- Streams text chunks progressively
- Manages conversation state

### Problem 2: User Data Isolation

**Challenge**: Ensuring complete separation of user documents in a multi-tenant system.

**Solution**: Implemented namespace-based isolation:
- Each user gets a unique Pinecone namespace
- JWT tokens carry user identity
- All retrieval operations scoped to user namespace
- Namespace existence validation before queries

### Problem 3: Answer Quality & Hallucination

**Challenge**: LLMs can generate plausible but incorrect answers.

**Solution**: Multi-agent verification pipeline:
- Context critic filters irrelevant chunks
- Summarization agent cites sources
- Verification agent cross-checks facts
- Reasoning model for final validation

### Problem 4: Complex Query Understanding

**Challenge**: Users ask complex, multi-part questions.

**Solution**: Query planning agent that:
- Decomposes complex queries
- Creates structured search strategy
- Generates sub-questions
- Improves retrieval accuracy

