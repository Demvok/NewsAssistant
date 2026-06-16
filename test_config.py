#!/usr/bin/env python
"""Quick test to verify configuration system works correctly."""

import sys

try:
    from config import settings
    
    print("✓ Configuration loaded successfully!")
    print("\n📋 Configuration Summary:")
    print(f"  Application: {settings.app_name} v{settings.app_version}")
    print(f"\n  LM Studio Configuration:")
    print(f"    - Base URL: {settings.lm_studio_base_url}")
    print(f"    - Chat Model: {settings.chat_model}")
    print(f"    - Embedding Model: {settings.embedding_model}")
    print(f"    - Request Timeout: {settings.request_timeout}s")
    print(f"\n  Database Configuration:")
    print(f"    - URL configured: {'YES' if settings.database_url else 'NO'}")
    print(f"\n  Generation Settings:")
    print(f"    - Temperature: {settings.temperature}")
    print(f"    - Top-P: {settings.top_p}")
    print(f"    - Top-K: {settings.top_k}")
    print(f"    - Max Tokens: {settings.max_tokens}")
    print(f"\n  RAG Configuration:")
    print(f"    - Enabled by default: {settings.rag_enabled_default}")
    print(f"    - Chunk size: {settings.chunk_size}")
    print(f"    - Chunk overlap: {settings.chunk_overlap}")
    print(f"    - Top-K retrieval: {settings.top_k_retrieval}")
    print(f"\n  Application Settings:")
    print(f"    - Multi-Agent Enabled: {settings.multi_agent_enabled_default}")
    print(f"    - Log Level: {settings.log_level}")
    print(f"    - Theme: {settings.streamlit_theme}")
    print(f"\n  Directories:")
    print(f"    - Project Root: {settings.project_root}")
    print(f"    - Data Directory: {settings.data_dir}")
    print(f"    - ChromaDB: {settings.project_root / settings.chroma_db_dir}")
    print(f"    - Logs: {settings.logs_dir}")
    
    print("\n✅ All configuration sources loaded successfully!")
    print("   - config.yaml (non-sensitive settings)")
    print("   - .env (sensitive data)")
    
    sys.exit(0)
    
except Exception as e:
    print(f"❌ Configuration Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
