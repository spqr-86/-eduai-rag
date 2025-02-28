from typing import List, Optional, Dict, Any, Type, Union
import logging
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Управляет созданием и работой с векторными хранилищами."""
    
    @staticmethod
    def create_chroma_store(
        embedding_function: Embeddings,
        persist_directory: Optional[str] = None,
        collection_name: str = "eduai_documents",
        **kwargs
    ) -> Chroma:
        """
        Создает или открывает хранилище ChromaDB.
        
        Args:
            embedding_function: Функция для создания эмбеддингов
            persist_directory: Директория для сохранения данных
            collection_name: Имя коллекции
            **kwargs: Дополнительные параметры для ChromaDB
            
        Returns:
            Chroma: Экземпляр векторного хранилища
        """
        client_settings = kwargs.pop('client_settings', None)
        
        if persist_directory:
            # Создаем директорию, если она не существует
            os.makedirs(persist_directory, exist_ok=True)
        
        return Chroma(
            embedding_function=embedding_function,
            persist_directory=persist_directory,
            collection_name=collection_name,
            client_settings=client_settings,
            **kwargs
        )
    
    @staticmethod
    def create_faiss_store(
        embedding_function: Embeddings,
        index_name: str = "eduai_index",
        **kwargs
    ) -> FAISS:
        """
        Создает векторное хранилище FAISS.
        
        Args:
            embedding_function: Функция для создания эмбеддингов
            index_name: Имя индекса
            **kwargs: Дополнительные параметры для FAISS
            
        Returns:
            FAISS: Экземпляр векторного хранилища FAISS
        """
        return FAISS(
            embedding_function=embedding_function,
            index_name=index_name,
            **kwargs
        )
    
    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding_function: Embeddings,
        store_type: str = "chroma",
        **kwargs
    ) -> VectorStore:
        """
        Создает векторное хранилище на основе списка документов.
        
        Args:
            documents: Список документов для индексации
            embedding_function: Функция для создания эмбеддингов
            store_type: Тип хранилища ('chroma', 'faiss')
            **kwargs: Дополнительные параметры для хранилища
            
        Returns:
            VectorStore: Экземпляр векторного хранилища
            
        Raises:
            ValueError: Если указан неподдерживаемый тип хранилища
        """
        store_factories = {
            "chroma": lambda docs, emb, **kw: Chroma.from_documents(
                documents=docs, 
                embedding=emb, 
                collection_name=kw.get('collection_name', 'eduai_documents'),
                persist_directory=kw.get('persist_directory'),
                **{k: v for k, v in kw.items() if k not in ['collection_name', 'persist_directory']}
            ),
            "faiss": lambda docs, emb, **kw: FAISS.from_documents(
                documents=docs, 
                embedding=emb, 
                **kw
            )
        }
        
        if store_type not in store_factories:
            supported_types = ", ".join(store_factories.keys())
            raise ValueError(
                f"Неподдерживаемый тип хранилища: {store_type}. "
                f"Поддерживаемые типы: {supported_types}"
            )
        
        # Фильтруем сложные метаданные, которые могут вызвать проблемы при сохранении
        clean_documents = filter_complex_metadata(documents)
        
        store = store_factories[store_type](clean_documents, embedding_function, **kwargs)
        logger.info(
            f"Создано хранилище {store_type} с {len(clean_documents)} документами"
        )
        
        return store
    
    @staticmethod
    def optimize_metadata(documents: List[Document]) -> List[Document]:
        """
        Оптимизирует метаданные документов для эффективного хранения.
        
        Args:
            documents: Список документов
            
        Returns:
            List[Document]: Документы с оптимизированными метаданными
        """
        optimized_docs = []
        
        # Сохраняем только полезные метаданные, избегая слишком больших объектов
        for doc in documents:
            # Копируем только нужные и безопасные для сериализации метаданные
            safe_metadata = {}
            
            if doc.metadata:
                # Список безопасных полей метаданных для хранения
                safe_fields = [
                    'source', 'title', 'author', 'page', 'chunk_id', 
                    'chunk_index', 'course_id', 'section', 'subsection',
                    'processed_at', 'url', 'category'
                ]
                
                for field in safe_fields:
                    if field in doc.metadata:
                        safe_metadata[field] = doc.metadata[field]
            
            # Создаем новый документ с оптимизированными метаданными
            optimized_doc = Document(
                page_content=doc.page_content,
                metadata=safe_metadata
            )
            optimized_docs.append(optimized_doc)
        
        return optimized_docs