from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str

    # Database
    database_url: str

    # Model settings
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o"

    # RAG pipeline settings
    confidence_threshold: float = 0.75
    top_k_chunks: int = 20       # how many chunks to fetch from each retriever
    rerank_top_n: int = 5        # how many chunks to pass to LLM after reranking

    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 50

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # docker-compose vars like POSTGRES_USER live in .env too — ignore them
    )


# Single instance imported everywhere in the app
# Python caches module imports so this is only created once
settings = Settings()
