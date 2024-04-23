# Data Agents with LlamaIndex

## AIM

- Understand the LlamaIndex framework
- Build indexes for both qualitative and quantiative data, leveraging metadata filtering
    - semantic pipeline
    - SQL-based pipeline
- Build an agentic RAG system to learn what the best adventure movies are

## Context Augmentation = RAG

- RAG = Dense Vector Retrieval + In-Context Learning
- Dense Vector Retrieval
- In-Context Learning

## LlamaIndex v0.10

- llama-index-core : main abstractions and components
- LlamaHub : plugins
- Core concepts:

    (1) Loading : Nodes
    (2) Indexing : Embeddings
    (3) Storing : Indices
    (4) Querying : Retrievers
    (5) Evaluation : RAGAS

### Loading: Nodes

- Chunking of source document using Node Parser
- Stores metadata

### Indexing: Embedding

- Indexing = structuring data for easy retrieval
- Embeddings: split docs into chunks (nodes), and create embeddings for each chunk

### Storage

- storing = avoid re-indexing

### Querying (retrievers)

- Querying = asking questions by leveraging LLMs and data structures
- Retrievers: fetches the most relevant context (nodes) given a query
- Response Synthesis: combines retrieved nodes
- Advanced RAG: https://www.youtube.com/watch?v=xmfPh1Fv2kk

### Query Engines

- generic interface that allows you to ask questions over your data
- built on one or many indexes via retrievers

## Data Agents

- LLM-powered knowledge workers
- perform automated search & retrieval
- calling external service API, processing response, storing it for later
- OpenAI Function Agent

    - building a data agent requires the following core ocmponents:
        (1) a reasoning loop
        (2) tool abstractions

## Auto Retriever Functional Tool

- Steps:
    - look at the query
    - look at the metadata
    - select the correct metadata filter
    - query the filtered index

- combined between (1) semantic data (qualitative), and (2) SQL-based (quantitative)



