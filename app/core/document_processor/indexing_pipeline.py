from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path
import os
import time

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from app.core.document_processor.document_loader import DocumentLoader
from app.core.document_processor.document_splitter import DocumentSplitter
from app.core.document_processor.metadata_enricher import MetadataEnricher
from app.core.embeddings.embedding_factory import EmbeddingFactory
from app.core.vectorstores.vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)

class DocumentIndexingPipeline:
    """
    Конвейер для индексации документов от загрузки до хранения в векторной БД.
    Реализует принцип SRP из SOLID, инкапсулируя процесс индексации документов.
    """
    
    def __init__(
        self,
        embedding_provider: Optional[Embeddings] = None,
        embedding_config: Optional[Dict[str, Any]] = None,
        vector_store_type: str = "chroma",
        vector_store_config: Optional[Dict[str, Any]] = None
    ):
        """
        Инициализирует конвейер индексации.
        
        Args:
            embedding_provider: Готовый провайдер эмбеддингов или None для создания нового
            embedding_config: Конфигурация для создания провайдера эмбеддингов
            vector_store_type: Тип векторного хранилища
            vector_store_config: Конфигурация для векторного хранилища
        """
        # Инициализируем провайдер эмбеддингов
        if embedding_provider is None:
            embedding_config = embedding_config or {"provider": "openai"}
            self.embedding_provider = EmbeddingFactory.create_embeddings(**embedding_config)
        else:
            self.embedding_provider = embedding_provider
        
        # Сохраняем настройки векторного хранилища
        self.vector_store_type = vector_store_type
        self.vector_store_config = vector_store_config or {}
    
    def process_file(
        self,
        file_path: Union[str, Path],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        splitter_type: str = "recursive"
    ) -> List[Document]:
        """
        Обрабатывает один файл для индексации.
        
        Args:
            file_path: Путь к файлу
            chunk_size: Размер чанка для разбиения
            chunk_overlap: Размер перекрытия
            splitter_type: Тип разделителя
            
        Returns:
            List[Document]: Подготовленные документы
        """
        start_time = time.time()
        logger.info(f"Начало обработки файла {file_path}")
        
        # 1. Загрузка документа
        documents = DocumentLoader.load_document(file_path)
        logger.info(f"Загружено {len(documents)} фрагментов")
        
        # 2. Разбиение на чанки
        split_docs = DocumentSplitter.split_documents(
            documents,
            splitter_type=splitter_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        logger.info(f"Документ разбит на {len(split_docs)} чанков")
        
        # 3. Обогащение метаданными
        enriched_docs = MetadataEnricher.add_chunk_metadata(split_docs)
        
        # 4. Оптимизация метаданных для хранения
        optimized_docs = VectorStoreManager.optimize_metadata(enriched_docs)
        
        processing_time = time.time() - start_time
        logger.info(f"Файл обработан за {processing_time:.2f} секунд")
        
        return optimized_docs
    
    def process_directory(
        self,
        dir_path: Union[str, Path],
        recursive: bool = True,
        **kwargs
    ) -> List[Document]:
        """
        Обрабатывает все поддерживаемые файлы в директории.
        
        Args:
            dir_path: Путь к директории
            recursive: Обрабатывать ли вложенные директории
            **kwargs: Дополнительные параметры для process_file
            
        Returns:
            List[Document]: Подготовленные документы
        """
        start_time = time.time()
        logger.info(f"Начало обработки директории {dir_path}")
        
        # 1. Загрузка всех документов
        documents = DocumentLoader.load_directory(dir_path, recursive=recursive)
        logger.info(f"Загружено {len(documents)} фрагментов из директории")
        
        # 2. Разбиение на чанки
        chunk_size = kwargs.get('chunk_size', 1000)
        chunk_overlap = kwargs.get('chunk_overlap', 200)
        splitter_type = kwargs.get('splitter_type', 'recursive')
        
        split_docs = DocumentSplitter.split_documents(
            documents,
            splitter_type=splitter_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        logger.info(f"Документы разбиты на {len(split_docs)} чанков")
        
        # 3. Обогащение метаданными
        enriched_docs = MetadataEnricher.add_chunk_metadata(split_docs)
        
        # 4. Оптимизация метаданных для хранения
        optimized_docs = VectorStoreManager.optimize_metadata(enriched_docs)
        
        processing_time = time.time() - start_time
        logger.info(f"Директория обработана за {processing_time:.2f} секунд")
        
        return optimized_docs
    
    def create_vector_store(
        self,
        documents: List[Document],
        persist_directory: Optional[str] = None
    ) -> VectorStore:
        """
        Создает векторное хранилище из подготовленных документов.
        
        Args:
            documents: Список подготовленных документов
            persist_directory: Директория для сохранения (для поддерживаемых хранилищ)
            
        Returns:
            VectorStore: Созданное векторное хранилище
        """
        # Объединяем пользовательские настройки с переданными параметрами
        store_config = self.vector_store_config.copy()
        
        if persist_directory:
            store_config['persist_directory'] = persist_directory
        
        # Создаем векторное хранилище
        vector_store = VectorStoreManager.from_documents(
            documents=documents,
            embedding_function=self.embedding_provider,
            store_type=self.vector_store_type,
            **store_config
        )
        
        return vector_store
    
    def full_indexing_pipeline(
        self,
        source_path: Union[str, Path],
        persist_directory: Optional[str] = None,
        is_directory: bool = False,
        **kwargs
    ) -> VectorStore:
        """
        Выполняет полный процесс индексации от загрузки до создания хранилища.
        
        Args:
            source_path: Путь к файлу или директории
            persist_directory: Директория для сохранения хранилища
            is_directory: Указывает, является ли source_path директорией
            **kwargs: Дополнительные параметры для обработки
            
        Returns:
            VectorStore: Созданное векторное хранилище
        """
        start_time = time.time()
        logger.info(f"Начало полного процесса индексации для {source_path}")
        
        # 1. Обработка документов
        if is_directory:
            documents = self.process_directory(source_path, **kwargs)
        else:
            documents = self.process_file(source_path, **kwargs)
        
        # 2. Создание векторного хранилища
        vector_store = self.create_vector_store(documents, persist_directory)
        
        # 3. Сохранение хранилища (если поддерживается)
        if hasattr(vector_store, 'persist') and persist_directory:
            vector_store.persist()
            logger.info(f"Векторное хранилище сохранено в {persist_directory}")
        
        processing_time = time.time() - start_time
        logger.info(
            f"Полный процесс индексации завершен за {processing_time:.2f} секунд. "
            f"Обработано {len(documents)} документов."
        )
        
        return vector_store
