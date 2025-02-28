from typing import List, Dict, Any, Optional, Callable
import logging
from datetime import datetime
import hashlib

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class MetadataEnricher:
    """Обогащает метаданные документов дополнительной информацией."""
    
    @staticmethod
    def add_chunk_metadata(
        documents: List[Document],
        add_chunk_id: bool = True,
        add_timestamp: bool = True
    ) -> List[Document]:
        """
        Добавляет базовые метаданные к чанкам документов.
        
        Args:
            documents: Список документов для обработки
            add_chunk_id: Добавлять ли уникальный ID для чанка
            add_timestamp: Добавлять ли временную метку обработки
            
        Returns:
            List[Document]: Документы с обогащенными метаданными
        """
        timestamp = datetime.now().isoformat()
        
        for i, doc in enumerate(documents):
            # Инициализируем метаданные, если их нет
            if doc.metadata is None:
                doc.metadata = {}
                
            # Добавляем индекс чанка в любом случае
            doc.metadata['chunk_index'] = i
            
            # Добавляем уникальный ID
            if add_chunk_id:
                # Создаем хеш на основе содержимого и позиции
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:10]
                doc.metadata['chunk_id'] = f"chunk_{i}_{content_hash}"
            
            # Добавляем временную метку
            if add_timestamp:
                doc.metadata['processed_at'] = timestamp
        
        return documents
    
    @staticmethod
    def extract_course_metadata(
        documents: List[Document],
        course_id_pattern: Optional[Callable[[str], Optional[str]]] = None
    ) -> List[Document]:
        """
        Извлекает и добавляет метаданные о курсе из пути к файлу.
        
        Args:
            documents: Список документов
            course_id_pattern: Функция для извлечения ID курса из пути
            
        Returns:
            List[Document]: Документы с метаданными курса
        """
        for doc in enumerate(documents):
            if 'source' in doc.metadata:
                source_path = doc.metadata['source']
                
                # Извлекаем информацию о курсе из пути
                if course_id_pattern and callable(course_id_pattern):
                    course_id = course_id_pattern(source_path)
                    if course_id:
                        doc.metadata['course_id'] = course_id
        
        return documents
    
    @staticmethod
    def custom_metadata_processor(
        documents: List[Document],
        processor_fn: Callable[[Document], Document]
    ) -> List[Document]:
        """
        Применяет пользовательскую функцию для обработки метаданных.
        
        Args:
            documents: Список документов
            processor_fn: Пользовательская функция обработки
            
        Returns:
            List[Document]: Обработанные документы
        """
        processed_docs = []
        
        for doc in documents:
            processed_doc = processor_fn(doc)
            processed_docs.append(processed_doc)
        
        return processed_docs