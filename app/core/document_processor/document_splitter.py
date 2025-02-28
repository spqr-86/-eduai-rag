from typing import List, Dict, Any, Optional
import logging

from langchain_core.documents import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter,
    HTMLHeaderTextSplitter
)

logger = logging.getLogger(__name__)

class DocumentSplitter:
    """Класс для разбиения документов на чанки оптимального размера."""
    
    @staticmethod
    def create_recursive_splitter(
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ) -> RecursiveCharacterTextSplitter:
        """
        Создает оптимизированный разделитель, который учитывает структуру текста.
        
        Args:
            chunk_size: Максимальный размер чанка
            chunk_overlap: Размер перекрытия между соседними чанками
            separators: Список разделителей в порядке приоритета
            
        Returns:
            RecursiveCharacterTextSplitter: Настроенный разделитель
        """
        default_separators = ["\n\n", "\n", ".", " ", ""]
        separators = separators or default_separators
        
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators
        )
    
    @staticmethod
    def create_token_splitter(
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base"  # openai's gpt-4 encoding
    ) -> TokenTextSplitter:
        """
        Создает разделитель на основе токенов, оптимальный для использования с LLM.
        
        Args:
            chunk_size: Максимальное количество токенов в чанке
            chunk_overlap: Количество перекрывающихся токенов
            encoding_name: Название кодировки токенизатора
            
        Returns:
            TokenTextSplitter: Настроенный разделитель токенов
        """
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name
        )
    
    @staticmethod
    def split_documents(
        documents: List[Document],
        splitter_type: str = "recursive",
        **splitter_kwargs
    ) -> List[Document]:
        """
        Разбивает список документов на чанки оптимального размера.
        
        Args:
            documents: Список документов для разбиения
            splitter_type: Тип разделителя ('recursive', 'token', 'markdown', 'html')
            **splitter_kwargs: Дополнительные параметры для разделителя
            
        Returns:
            List[Document]: Список разбитых документов
            
        Raises:
            ValueError: Если указан неподдерживаемый тип разделителя
        """
        splitter_factories = {
            "recursive": DocumentSplitter.create_recursive_splitter,
            "token": DocumentSplitter.create_token_splitter,
            "markdown": lambda **kwargs: MarkdownTextSplitter(**kwargs),
            "html": lambda **kwargs: RecursiveCharacterTextSplitter(
                separators=["<head>", "<body>", "<h1>", "<h2>", "<h3>", "<p>", "\n\n", "\n", " ", ""],
                **kwargs
            )
        }
        
        if splitter_type not in splitter_factories:
            supported_types = ", ".join(splitter_factories.keys())
            raise ValueError(
                f"Неподдерживаемый тип разделителя: {splitter_type}. "
                f"Поддерживаемые типы: {supported_types}"
            )
        
        # Создаем разделитель нужного типа с переданными параметрами
        splitter = splitter_factories[splitter_type](**splitter_kwargs)
        
        # Разбиваем документы
        split_docs = splitter.split_documents(documents)
        
        logger.info(
            f"Документы разбиты с использованием {splitter_type} разделителя: "
            f"{len(documents)} документов -> {len(split_docs)} фрагментов"
        )
        
        return split_docs