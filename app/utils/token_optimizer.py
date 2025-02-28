# app/utils/token_optimizer.py
from typing import List, Dict, Any, Optional, Union, Callable
import logging
import tiktoken
from collections import Counter

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class TokenOptimizer:
    """
    Утилита для оптимизации использования токенов в RAG-системе.
    """
    
    @staticmethod
    def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
        """
        Подсчитывает количество токенов в тексте.
        
        Args:
            text: Текст для подсчета токенов
            encoding_name: Имя кодировки токенизатора
            
        Returns:
            int: Количество токенов
        """
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        return len(tokens)
    
    @staticmethod
    def truncate_to_token_limit(
        text: str, 
        max_tokens: int = 8000,
        encoding_name: str = "cl100k_base"
    ) -> str:
        """
        Обрезает текст до указанного лимита токенов.
        
        Args:
            text: Исходный текст
            max_tokens: Максимальное количество токенов
            encoding_name: Имя кодировки
            
        Returns:
            str: Обрезанный текст
        """
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    
    @staticmethod
    def estimate_request_tokens(
        system_prompt: str,
        user_query: str,
        context_docs: List[Document],
        encoding_name: str = "cl100k_base"
    ) -> Dict[str, int]:
        """
        Оценивает количество токенов для запроса к LLM.
        
        Args:
            system_prompt: Системный промпт
            user_query: Запрос пользователя
            context_docs: Контекстные документы
            encoding_name: Имя кодировки
            
        Returns:
            Dict[str, int]: Количество токенов для разных компонентов
        """
        # Подсчет токенов для компонентов запроса
        system_tokens = TokenOptimizer.count_tokens(system_prompt, encoding_name)
        query_tokens = TokenOptimizer.count_tokens(user_query, encoding_name)
        
        # Подсчет токенов для контекста
        context_text = "\n\n".join(doc.page_content for doc in context_docs)
        context_tokens = TokenOptimizer.count_tokens(context_text, encoding_name)
        
        # Общее количество токенов
        total_tokens = system_tokens + query_tokens + context_tokens
        
        return {
            "system_tokens": system_tokens,
            "query_tokens": query_tokens,
            "context_tokens": context_tokens,
            "total_tokens": total_tokens
        }
    
    @staticmethod
    def filter_redundant_chunks(
        documents: List[Document],
        similarity_threshold: float = 0.85
    ) -> List[Document]:
        """
        Фильтрует избыточные чанки с похожим содержанием.
        Эта функция требует дополнительной библиотеки для вычисления сходства,
        например, scikit-learn или sentence-transformers.
        
        Args:
            documents: Список документов
            similarity_threshold: Порог сходства для фильтрации
            
        Returns:
            List[Document]: Отфильтрованные документы
        """
        # Здесь должна быть реализация на основе сравнения чанков
        # Для простоты примера просто возвращаем исходные документы
        
        return documents
    
    @staticmethod
    def rerank_by_relevance(
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Document]:
        """
        Переранжирует документы по релевантности к запросу.
        
        Args:
            query: Запрос пользователя
            documents: Список документов
            top_k: Количество документов для возврата
            
        Returns:
            List[Document]: Наиболее релевантные документы
        """
        # Здесь должна быть реализация переранжирования
        # Для простоты примера просто возвращаем первые top_k документов
        
        return documents[:min(top_k, len(documents))]
    
    @staticmethod
    def get_optimal_context_size(
        model_max_tokens: int = 16000,
        expected_output_tokens: int = 1000,
        system_prompt_tokens: int = 500,
        query_tokens: int = 100,
        buffer_tokens: int = 500
    ) -> int:
        """
        Вычисляет оптимальный размер контекста для модели.
        
        Args:
            model_max_tokens: Максимальное количество токенов для модели
            expected_output_tokens: Ожидаемое количество токенов в ответе
            system_prompt_tokens: Токены в системном промпте
            query_tokens: Токены в запросе пользователя
            buffer_tokens: Буферные токены для безопасности
            
        Returns:
            int: Оптимальный размер контекста в токенах
        """
        available_tokens = model_max_tokens - expected_output_tokens - system_prompt_tokens - query_tokens - buffer_tokens
        
        # Убедимся, что у нас есть положительное число токенов для контекста
        optimal_context_tokens = max(0, available_tokens)
        
        return optimal_context_tokens