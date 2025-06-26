# Qdrant Documentation Hybrid-Search Bot

A Discord bot that answers questions about Qdrant documentation using hybrid search (dense + sparse vectors) and OpenAI, powered by a custom-ingested Qdrant collection.

---

## Features

- **Hybrid Search**: Combines dense (BAAI/bge-large-en-v1.5) and sparse (Qdrant/minicoil-v1) embeddings with RRF fusion.
- **Streaming Ingestion**: Scrapes, chunks, deduplicates, embeds, and uploads Qdrant docs to your Qdrant instance.
- **Discord Bot**: Answers questions in threads, using OpenAI for responses and remembering thread context.

---

## Setup

1. **Clone and install dependencies:**
    ```bash
    git clone <your-repo-url>
    cd qdrant-bot
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

2. **Configure environment:**
    - Create a `.env` file in the root directory:
      ```
      QDRANT_CLOUD_URL=your_qdrant_url
      QDRANT_CLOUD_API_KEY=your_qdrant_api_key
      DISCORD_BOT_TOKEN=your_discord_bot_token
      OPENAI_API_KEY=your_openai_api_key
      ```

---

## Usage

- **Ingest documentation:**
    ```bash
    python ingest_data.py
    ```
- **Start the Discord bot:**
    ```bash
    python bot.py
    ```

---

