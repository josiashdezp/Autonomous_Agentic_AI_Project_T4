# 📘 Travel RAG Module (Updated)

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/RAG-Travel%20Knowledge-green" alt="RAG">
  <img src="https://img.shields.io/badge/Vector%20DB-Chroma-purple" alt="Chroma">
  <img src="https://img.shields.io/badge/Embeddings-OpenAI-orange" alt="OpenAI Embeddings">
  <img src="https://img.shields.io/badge/Status-In%20Development-yellow" alt="Status">
</p>

This module implements the Retrieval-Augmented Generation (RAG) layer for a travel-focused AI agent. It collects curated
travel content, structures it, splits it semantically, generates embeddings, and stores it in a Chroma vector database
for efficient retrieval.

The system is designed as an **offline-first indexing pipeline** that integrates into a larger **agentic AI architecture
using LangGraph**.

---

## ⚠️ REQUIRED SETUP (IMPORTANT)

### 1. Environment Variables

You MUST create a `.env` file in the root of the project with:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GIT_HUB_JHP=your_key_here
```

⚠️ Without these keys:

* embeddings will fail
* observability will not work
* the system will not run correctly

---

### 2. Vector Database Setup

You MUST do one of the following:

### Option A — Use Prebuilt DB (Recommended)

1. Locate:

```text
chroma_travel_db.zip
```

2. Extract it into:

```text
/rag/
```

Final structure:

```text
/rag/chroma_travel_db/
    chroma.sqlite3
    <uuid folders>
```

⚠️ If this is not done, the RAG will return empty results.

---

### Option B — Rebuild the DB

```bash
python rag/build_rag_index.py
```

This will:

* scrape sources
* process documents
* generate embeddings
* build the vector DB

⚠️ Requires valid OpenAI API key and may take time.

---

## Table of Contents

* [Overview](#overview)
* [What This Module Does](#what-this-module-does)
* [Architecture](#architecture)
* [Data Sources](#data-sources)
* [Project Structure](#project-structure)
* [How the Pipeline Works](#how-the-pipeline-works)
* [Installation](#installation)
* [Usage](#usage)
* [Retrieval Layer](#retrieval-layer)
* [Design Notes](#design-notes)
* [Future Improvements](#future-improvements)

---

## Overview

This module provides a complete RAG pipeline for travel knowledge. It ingests curated content, converts it into
structured objects, splits it into semantic sections, chunks it for embeddings, and stores it in a vector database.

The dataset is built offline and reused during inference, enabling fast and stable retrieval.

---

## What This Module Does

* Ingests travel content from curated sources
* Normalizes content into `TravelDocument` objects
* Splits documents into semantic `TravelSection` units
* Converts sections into embedding-ready chunks
* Stores embeddings in a persistent Chroma vector database
* Enables semantic retrieval for downstream agents

---

## Architecture

```text
SOURCE_REGISTRY
        ↓
Ingestors
        ↓
TravelDocument
        ↓
Text Cleaning
        ↓
SectionSplitter
        ↓
TravelSection
        ↓
Chunking (LangChain)
        ↓
Embeddings (OpenAI)
        ↓
Chroma Vector Store
        ↓
Retrieval Service
```

```mermaid
flowchart TD

    A[SOURCE_REGISTRY<br>rag_config.py] --> B[Ingestors<br>ingestors.py]

    B --> C[TravelDocument<br>structures.py]

    C --> D[Text Cleaning<br>TravelTextCleaner]

    D --> E[SectionSplitter<br>splitters.py]

    E --> F[TravelSection]

    F --> G[Chunking<br>LangChain Splitter]

    G --> H[Embeddings<br>OpenAI]

    H --> I[Chroma Vector DB<br>chroma_travel_db]

    I --> J[Retrieval Layer<br>TravelRAGService]

    J --> K[LangGraph Agent]

---

## Data Sources

Sources are defined through a centralized registry:

```python
SOURCE_REGISTRY
```

This controls:

* which sources are enabled
* how URLs are constructed
* scraping depth
* parser (ingestor) class

Examples include:

* Wikivoyage (depth-1 scraping)
* VisitTheUSA
* Recreation.gov
* The Dyrt
* Expatistan
* GasBuddy

Some sources are disabled due to scraping restrictions (e.g., AllTrails).

---

## Project Structure

```text
rag/
├── README.md
├── rag_config.py
├── ingestors.py
├── indexing.py
├── splitters.py
├── structures.py
├── build_rag_index.py
└── chroma_travel_db/   (generated or extracted)
```

### File Responsibilities

**structures.py** → data models (`TravelDocument`, `TravelSection`)
**ingestors.py** → scraping + parsing logic for each source
**splitters.py** → semantic sectioning logic
**indexing.py** → cleaning, chunking, embedding, batching
**rag_config.py** → source registry + configuration
**build_rag_index.py** → orchestrates full pipeline

---

## How the Pipeline Works

### 1. Ingestion

Each source is processed through an ingestor class derived from `BaseHTMLIngestor`.

The system supports:

* generic article parsing
* custom parsing (e.g., VisitTheUSA)
* depth-1 scraping (Wikivoyage)

---

### 2. Cleaning

Text is normalized to remove noise:

```python
TravelTextCleaner
```

---

### 3. Semantic Sectioning

Documents are split using a heading-based heuristic:

```python
SectionSplitter.split_document(...)
```

This creates structured sections instead of naive chunks.

---

### 4. Chunking

Sections are converted into LangChain `Document` objects and split using:

```python
RecursiveCharacterTextSplitter
```

---

### 5. Embedding & Indexing

Chunks are embedded using:

```python
text-embedding-3-small
```

and stored in Chroma.

Batching is used to improve stability:

```python
build_index(batch_size=50)
```

---

### 6. Retrieval

The vector DB supports similarity search and can be integrated into a retrieval service or agent pipeline.

---

## Installation

```bash
pip install beautifulsoup4 requests langchain langchain-openai langchain-chroma langchain-text-splitters colorama python-dotenv
```

---

## Usage

### Build the Index

```bash
python rag/build_rag_index.py
```

---

### Query the RAG

```python
from langchain_chroma import Chroma
from rag.service import TravelRAGService

vector_store = Chroma(
    collection_name="us_travel_rag",
    persist_directory="./chroma_travel_db",
    embedding_function=your_embeddings
)

rag = TravelRAGService(vector_store)

result = rag.search(
    query="best places to visit",
    destination="New York City"
)
```

---

## Retrieval Layer

The retrieval layer wraps similarity search and returns structured results with metadata such as:

* destination
* state
* source
* category

This layer is designed to be used inside a **LangGraph agent node**.

---

## Design Notes

* Registry-driven ingestion for scalability
* Structured schemas instead of raw text
* Separation of ingestion and indexing
* Offline-first approach for performance
* Batch indexing to avoid API failures
* Modular design aligned with agentic workflows

---

## Future Improvements

* Hybrid retrieval (keyword + vector)
* Reranking models
* Query rewriting
* More structured datasets
* Evaluation framework

---

## Author

Part of a broader project focused on autonomous AI agents with reasoning, tools, and retrieval.

Hernandez Perez, Josias
[https://github.com/josiashdezp](https://github.com/josiashdezp)

---

## Final Note

The dataset used by this RAG is generated from a structured locations file:

```text
data/usa_locations.json
```

which defines all states and major cities used for ingestion. 