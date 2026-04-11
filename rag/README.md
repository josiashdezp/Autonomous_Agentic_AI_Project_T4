# Travel RAG Module

<p align="left">

&#x20; <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">

&#x20; <img src="https://img.shields.io/badge/RAG-Travel%20Knowledge-green" alt="RAG">

&#x20; <img src="https://img.shields.io/badge/Vector%20DB-Chroma-purple" alt="Chroma">

&#x20; <img src="https://img.shields.io/badge/Embeddings-OpenAI-orange" alt="OpenAI Embeddings">

&#x20; <img src="https://img.shields.io/badge/Status-In%20Development-yellow" alt="Status">

</p>

This module implements the Retrieval-Augmented Generation (RAG) layer for a travel-focused AI agent. It is responsible for collecting curated travel content, transforming it into structured documents, splitting content into semantically meaningful sections, generating embeddings, and storing the resulting chunks in a Chroma vector database for retrieval at runtime.
The design separates offline indexing from online retrieval. This makes the system easier to maintain, extend, and integrate into larger agentic workflows.

\---



## Table of Contents



*[Overview](#overview)

*[What This Module Does](#what-this-module-does)

*[Architecture](#architecture)

*[Data Sources](#data-sources)

*[Project Structure](#project-structure)

*[How the Pipeline Works](#how-the-pipeline-works)

*[Installation](#installation)

*[Usage](#usage)

*[Retrieval Layer](#retrieval-layer)

*[Design Notes](#design-notes)

*[Future Improvements](#future-improvements)



\---



## Overview



This module provides an end-to-end RAG pipeline for travel knowledge. It ingests curated content from external sources, standardizes it into a common schema, cleans and sections the text, chunks it into embedding-ready units, and stores those chunks in a vector database.



At runtime, the retrieval service performs similarity search and can optionally filter results by destination.



\---



## What This Module Does



*Ingests travel content from multiple curated sources

*Normalizes raw content into structured `TravelDocument` objects

*Splits documents into semantic `TravelSection` units

*Chunks sections into embedding-sized LangChain documents

*Stores embeddings in a persistent Chroma collection

*Exposes a retrieval service for semantic search



\---



## Architecture



```text

Curated Source Registry

&#x20;       ↓

Source Ingestors

&#x20;       ↓

TravelDocument

&#x20;       ↓

Text Cleaning

&#x20;       ↓

SectionSplitter

&#x20;       ↓

TravelSection

&#x20;       ↓

Chunking

&#x20;       ↓

Embeddings

&#x20;       ↓

Chroma Vector Store

&#x20;       ↓

TravelRAGService

```



\---



## Data Sources



This module uses curated data instead of uncontrolled web crawling. The dataset is defined through a structured source registry that specifies what content should be ingested.



The pipeline currently integrates:



*Wikipedia

*National Park Service (NPS)

*Visit The USA



Each source is processed through a dedicated ingestor that fetches, cleans, and transforms content into a standardized format. This ensures consistency across heterogeneous sources and maintains high data quality.



This approach ensures:



*Data quality and consistency

*Reproducibility of the dataset

*Full control over sources



\---



## Project Structure



```text

rag/

├── README.md

├── ingestors.py

├── indexing.py

├── service.py

├── splitters.py

└── structures.py

```



### File Responsibilities



**\*structures.py\**— defines core data models (`TravelDocument`, `TravelSection`)

**\*ingestors.py\**— handles data fetching and parsing for each source

**\*splitters.py\**— splits documents into semantic sections

**\*indexing.py\**— handles cleaning, chunking, and embedding preparation

**\*service.py\**— provides the retrieval interface



\---



## How the Pipeline Works



### 1. Ingestion



Content is collected from a curated registry. Each source-specific ingestor fetches and converts raw content into a standardized `TravelDocument` format with consistent metadata.



### 2. Cleaning



Text is normalized to remove noise such as extra whitespace and formatting inconsistencies.



### 3. Semantic Sectioning



Documents are split into logical sections using heading-based heuristics instead of naive chunking.



### 4. Chunking



Each section is broken into smaller chunks suitable for embedding, while preserving metadata.



### 5. Embedding \& Indexing



Chunks are embedded using OpenAI embeddings and stored in a Chroma vector database.



### 6. Retrieval



Similarity search retrieves relevant chunks based on user queries, optionally filtered by destination.



\---



## Installation



```bash

pip install beautifulsoup4 requests langchain langchain-openai langchain-chroma langchain-text-splitters colorama

```



Set your environment variable:



```bash

export OPENAI\_API\_KEY=your\_api\_key

```



Windows PowerShell:



```powershell

$env:OPENAI\_API\_KEY="your\_api\_key"

```



\---



## Usage



### Build the Index



```bash

python build\_rag\_index.py

```



This will:



*Fetch data from configured sources

*Process and clean documents

*Generate embeddings

*Store them in the Chroma vector database



\---



### Query the RAG Service



```python

from langchain\_chroma import Chroma

from rag.service import TravelRAGService



vector\_store = Chroma(

&#x20;   collection\_name="your\_collection",

&#x20;   persist\_directory="your\_directory",

&#x20;   embedding\_function=your\_embeddings

)



rag = TravelRAGService(vector\_store)



result = rag.search(

&#x20;   query="best landmarks to visit",

&#x20;   destination="New York City"

)



print(result)

```



\---



## Retrieval Layer



The retrieval layer wraps vector similarity search and formats results with source and destination context. It is intentionally minimal so that higher-level agent components can handle reasoning and response generation.



\---



## Design Notes



*Uses structured schemas instead of raw text

*Preserves metadata across all stages

*Separates indexing from retrieval

*Uses curated data sources instead of uncontrolled scraping

*Designed for integration into agent-based systems



\---



## Future Improvements



*Hybrid retrieval (vector + keyword search)

*Reranking models for improved accuracy

*Query rewriting and reasoning chains

*Additional data sources

*Evaluation framework for retrieval quality



\---



## Author



Part of a broader project focused on building autonomous AI agents with structured reasoning, tools, and retrieval capabilities.

Hernandez Perez, Josias - https://github.com/josiashdezp

Sen, Indu - https://github.com/indu-sen

Gadde, Srichandrika - 





