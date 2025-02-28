from typing import List, Dict, Any, Optional, Union
import logging
import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class ChromaManager:
    """
    Утилита для управления ChromaDB хранилищем.
    """
    
    def __init__(
        self,
        persist_directory: str,
        embedding_function: Embeddings,
        collection_name: str = "eduai_documents"
    ):
        """
        Инициализирует менеджер ChromaDB.
        
        Args:
            persist_directory: Директория для хранения данных
            embedding_function: Функция для создания эмбеддингов
            collection_name: Имя коллекции
        """
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        
        # Создаем директорию для хранения, если она не существует
        os.makedirs(persist_directory, exist_ok=True)
        
        # Инициализируем клиент ChromaDB
        self.client_settings = Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        )
        
        # Инициализируем хранилище
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            collection_name=collection_name,
            client_settings=self.client_settings
        )
        
        logger.info(f"ChromaDB инициализирована в {persist_directory}")
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> None:
        """
        Добавляет документы в ChromaDB.
        
        Args:
            documents: Список документов для добавления
            batch_size: Размер батча для добавления документов
        """
        total_docs = len(documents)
        logger.info(f"Добавление {total_docs} документов в ChromaDB")
        
        # Добавляем документы пакетами для оптимизации
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            self.db.add_documents(batch)
            logger.info(f"Добавлен батч {i//batch_size + 1}, всего {min(i + batch_size, total_docs)}/{total_docs}")
        
        # Сохраняем изменения
        self.db.persist()
        logger.info(f"ChromaDB сохранена с {total_docs} документами")
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Выполняет поиск по векторному хранилищу.
        
        Args:
            query: Запрос для поиска
            k: Количество результатов
            filter: Фильтр для запроса
            
        Returns:
            List[Document]: Найденные документы
        """
        results = self.db.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
        
        logger.info(f"Найдено {len(results)} документов по запросу: {query}")
        return results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Получает статистику коллекции.
        
        Returns:
            Dict[str, Any]: Статистика коллекции
        """
        # Получаем доступ к низкоуровневому клиенту
        chroma_client = self.db._client
        collection = chroma_client.get_collection(self.collection_name)
        
        # Получаем количество документов
        count = collection.count()
        
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": self.persist_directory
        }
    
    def delete_collection(self) -> None:
        """
        Удаляет коллекцию.
        """
        # Получаем доступ к низкоуровневому клиенту
        chroma_client = self.db._client
        chroma_client.delete_collection(self.collection_name)
        logger.info(f"Коллекция {self.collection_name} удалена")