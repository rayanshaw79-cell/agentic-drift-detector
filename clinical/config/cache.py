import os
import logging
from langchain.globals import set_llm_cache
from langchain_community.cache import RedisSemanticCache
from langchain_google_genai import GoogleGenerativeAIEmbeddings

log = logging.getLogger(__name__)

def init_semantic_cache():
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    google_api_key = os.getenv("GEMINI_API_KEY")
    
    if not google_api_key:
        log.warning("GEMINI_API_KEY not set. Skipping semantic cache initialization.")
        return
        
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", 
            google_api_key=google_api_key
        )
        
        # Configure global semantic cache
        set_llm_cache(RedisSemanticCache(
            embedding=embeddings,
            redis_url=redis_url,
            score_threshold=0.95  # only hit cache if >95% similar
        ))
        log.info("Redis Semantic Cache initialized via models/text-embedding-004.")
    except Exception as e:
        log.warning(f"Failed to initialize semantic cache: {e}")
