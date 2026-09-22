from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """Application configuration loaded from environment variables / .env, with defaults for local dev."""
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql+asyncpg://postgres:legalrag@localhost:5432/legalrag"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    model_small: str = "llama3.2:3b"
    model_medium: str = "llama3.1:8b"
    model_embed: str = "nomic-embed-text"

    # Routing
    confidence_threshold: float = 0.75
    cluster_timeout_seconds: int = 600

    # Obsidian
    obsidian_vault_path: str = "/home/revan/Desktop/Dissertation/Dissertation"
    obsidian_subfolder: str = "LegalChat"
    obsidian_enabled: bool = True

    # Cluster
    cluster_ssh_alias: str = "warwick-cluster"
    cluster_jump_host: str = "warwick-remote"
    cluster_user: str = "u5754610"
    cluster_project_dir: str = "/dcs/pg25/u5754610/Desktop/test/legal_rag"
    cluster_shared_input: str = "/dcs/pg25/u5754610/Desktop/test/legal_rag/shared/infer_requests"
    cluster_shared_output: str = "/dcs/pg25/u5754610/Desktop/test/legal_rag/shared/infer_results"
    cluster_eval_output: str = "/dcs/pg25/u5754610/Desktop/test/legal_rag/shared/eval_results"
    slurm_account: str = "wmlg"
    local_temp_dir: str = "/tmp/legalrag_cluster"

    # Session / RAG
    default_user_id: str = "default"
    max_context_messages: int = 4
    sac_summary_max_chars: int = 150
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_top_k: int = 8


settings = Settings()
