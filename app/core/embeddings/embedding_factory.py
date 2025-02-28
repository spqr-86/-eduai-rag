from typing import Union, Optional, Dict, Any, List
import logging
import os

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    SentenceTransformerEmbeddings,
    HuggingFaceBgeEmbeddings,
    CohereEmbeddings
)
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

class EmbeddingFactory:
    """Фабрика для создания различных моделей эмбеддингов."""
    
    @staticmethod
    def create_openai_embeddings(
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        **kwargs
    ) -> OpenAIEmbeddings:
        """
        Создает провайдер эмбеддингов OpenAI.
        
        Args:
            model: Модель для эмбеддингов
            dimensions: Размерность эмбеддингов
            **kwargs: Дополнительные параметры для OpenAIEmbeddings
            
        Returns:
            OpenAIEmbeddings: Настроенный провайдер эмбеддингов
        """
        # Получаем API ключ из переменных окружения, если не указан явно
        api_key = kwargs.pop('api_key', os.environ.get("OPENAI_API_KEY"))
        
        return OpenAIEmbeddings(
            model=model,
            dimensions=dimensions,
            api_key=api_key,
            **kwargs
        )
    
    @staticmethod
    def create_local_embeddings(
        model_name: str = "all-MiniLM-L6-v2",
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[HuggingFaceEmbeddings, SentenceTransformerEmbeddings]:
        """
        Создает локальный провайдер эмбеддингов на основе моделей HuggingFace.
        
        Args:
            model_name: Имя модели HuggingFace/SentenceTransformers
            model_kwargs: Параметры для модели
            **kwargs: Дополнительные параметры для провайдера
            
        Returns:
            Экземпляр провайдера локальных эмбеддингов
        """
        model_kwargs = model_kwargs or {"device": "cpu"}
        
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            **kwargs
        )
    
    @staticmethod
    def create_bge_embeddings(
        model_name: str = "BAAI/bge-small-en-v1.5",
        **kwargs
    ) -> HuggingFaceBgeEmbeddings:
        """
        Создает провайдер эмбеддингов BGE, оптимизированный для поиска.
        
        Args:
            model_name: Имя модели BGE
            **kwargs: Дополнительные параметры
            
        Returns:
            HuggingFaceBgeEmbeddings: Провайдер BGE эмбеддингов
        """
        return HuggingFaceBgeEmbeddings(
            model_name=model_name,
            **kwargs
        )
    
    @classmethod
    def create_embeddings(
        cls,
        provider: str = "openai",
        **kwargs
    ) -> Embeddings:
        """
        Создает провайдер эмбеддингов на основе указанного типа.
        
        Args:
            provider: Тип провайдера ('openai', 'local', 'bge', 'cohere')
            **kwargs: Параметры для конкретного провайдера
            
        Returns:
            Embeddings: Провайдер эмбеддингов
            
        Raises:
            ValueError: Если указан неподдерживаемый провайдер
        """
        provider_factories = {
            "openai": cls.create_openai_embeddings,
            "local": cls.create_local_embeddings,
            "bge": cls.create_bge_embeddings,
            "cohere": lambda **kw: CohereEmbeddings(**kw)
        }
        
        if provider not in provider_factories:
            supported_providers = ", ".join(provider_factories.keys())
            raise ValueError(
                f"Неподдерживаемый провайдер эмбеддингов: {provider}. "
                f"Поддерживаемые провайдеры: {supported_providers}"
            )
        
        embeddings = provider_factories[provider](**kwargs)
        logger.info(f"Создан провайдер эмбеддингов типа {provider}")
        
        return embeddings