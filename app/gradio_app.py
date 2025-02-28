# app/gradio_app.py
import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv

import gradio as gr

from app.core.embeddings.embedding_factory import EmbeddingFactory
from app.utils.chroma_manager import ChromaManager
from app.utils.token_optimizer import TokenOptimizer
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
import tiktoken

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Константы
DEFAULT_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/chroma_db")
EMBEDDING_TYPE = os.getenv("EMBEDDING_TYPE", "openai")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
MAX_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1000"))

# Глобальные объекты (будут инициализированы позже)
embedding_provider = None
vector_store = None
qa_chain = None
token_count = 0

def initialize_rag_system(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Инициализирует компоненты RAG-системы.
    
    Args:
        db_path: Путь к векторной базе данных
    """
    global embedding_provider, vector_store, qa_chain
    
    # Инициализируем провайдер эмбеддингов
    embedding_provider = EmbeddingFactory.create_embeddings(provider=EMBEDDING_TYPE)
    
    # Инициализируем ChromaDB
    chroma_manager = ChromaManager(
        persist_directory=db_path,
        embedding_function=embedding_provider,
        collection_name="eduai_documents",
    )
    
    vector_store = chroma_manager.db
    
    # Создаем retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Создаем промпт
    prompt_template = ChatPromptTemplate.from_template("""
    Ты - помощник для платформы Open edX. Отвечай на вопросы по учебным материалам,
    используя только предоставленный контекст. Если информации недостаточно,
    признай это. Не выдумывай информацию.
    
    Контекст:
    {context}
    
    Вопрос: {question}
    """)
    
    # Инициализируем LLM
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.2,
        max_tokens=MAX_TOKENS
    )
    
    # Создаем QA-цепочку LangChain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
    
    logger.info(f"RAG-система инициализирована с базой знаний из {db_path}")

def answer_query(
    query: str,
    db_path: str = DEFAULT_DB_PATH,
    k: int = 5,
    temperature: float = 0.2
) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
    """
    Отвечает на запрос пользователя с использованием RAG.
    """
    global token_count, qa_chain, vector_store
    
    try:
        logger.info(f"Получен запрос: {query}")
        logger.info(f"Параметры: db_path={db_path}, k={k}, temperature={temperature}")
        
        # Инициализируем систему, если это необходимо
        if vector_store is None or qa_chain is None:
            logger.info("Инициализация RAG-системы...")
            initialize_rag_system(db_path)
        
        # Обновляем параметры поиска в существующем retriever
        qa_chain.retriever.search_kwargs["k"] = k
        
        # Обновляем температуру в существующей модели
        if hasattr(qa_chain, 'llm') and qa_chain.llm.temperature != temperature:
            qa_chain.llm.temperature = temperature
        
        # Измеряем токены в запросе
        query_tokens = TokenOptimizer.count_tokens(query)
        
        # Получаем ответ
        logger.info(f"Выполнение запроса через RAG...")
        start_time = time.time()
        result = qa_chain.invoke({"query": query})
        end_time = time.time()
        
        answer = result["result"]
        source_docs = result["source_documents"]
        
        logger.info(f"Получен ответ длиной {len(answer)} символов")
        logger.info(f"Найдено {len(source_docs)} источников")
        
        # Готовим информацию об источниках для отображения
        sources = []
        context_text = ""
        
        for i, doc in enumerate(source_docs, 1):
            # Собираем контекст для подсчета токенов
            context_text += doc.page_content + "\n\n"
            
            # Извлекаем метаданные для отображения
            metadata = doc.metadata or {}
            source_info = {
                "index": i,
                "title": metadata.get("title", "Документ без названия"),
                "source": metadata.get("source", "Неизвестный источник"),
                "chunk_id": metadata.get("chunk_id", f"chunk_{i}"),
                "relevance": "Высокая" if i <= 2 else "Средняя" if i <= 4 else "Низкая"
            }
            sources.append(source_info)
        
        # Подсчитываем токены
        answer_tokens = TokenOptimizer.count_tokens(answer)
        context_tokens = TokenOptimizer.count_tokens(context_text)
        
        # Обновляем счетчик токенов
        token_count += query_tokens + context_tokens + answer_tokens
        
        # Собираем статистику
        token_stats = {
            "query_tokens": query_tokens,
            "context_tokens": context_tokens,
            "answer_tokens": answer_tokens,
            "total_tokens": query_tokens + context_tokens + answer_tokens,
            "cumulative_tokens": token_count,
            "processing_time": f"{(end_time - start_time):.2f} сек"
        }
        
        return answer, sources, token_stats
    
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {str(e)}", exc_info=True)
        return f"Произошла ошибка: {str(e)}", [], {"error": str(e)}

def reset_token_counter():
    """Сбрасывает счетчик токенов."""
    global token_count
    token_count = 0
    return "Счетчик токенов сброшен."

def create_ui():
    """Создает интерфейс Gradio."""
    with gr.Blocks(title="EduAI - Интеллектуальный ассистент для Open edX") as demo:
        gr.Markdown("# EduAI - Интеллектуальный ассистент для Open edX")
        gr.Markdown("""
        Этот демонстрационный интерфейс показывает работу RAG-системы для ответов на вопросы
        по учебным материалам Open edX. Система использует LangChain, ChromaDB и OpenAI для
        поиска и генерации ответов.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="Задайте вопрос о платформе Open edX",
                    placeholder="Например: Как создать курс в Open edX?",
                    lines=3
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Задать вопрос", variant="primary")
                    clear_btn = gr.Button("Очистить")
                
                with gr.Accordion("Параметры", open=False):
                    db_path = gr.Textbox(
                        label="Путь к базе знаний",
                        value=DEFAULT_DB_PATH
                    )
                    k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Количество документов для поиска (k)"
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.2,
                        step=0.1,
                        label="Температура генерации"
                    )
                    reset_btn = gr.Button("Сбросить счетчик токенов")
                
                answer_output = gr.Textbox(
                    label="Ответ",
                    lines=10,
                    placeholder="Здесь появится ответ на ваш вопрос..."
                )
            
            with gr.Column(scale=1):
                with gr.Tab("Источники"):
                    sources_output = gr.JSON(label="Использованные источники")
                
                with gr.Tab("Статистика токенов"):
                    token_stats_output = gr.JSON(label="Статистика использования токенов")
        
        # Примеры запросов
        examples = [
            ["Что такое Open edX?"],
            ["Какие типы оценивания поддерживает Open edX?"],
            ["Как создать курс на платформе Open edX?"],
            ["Какие модули доступны для создания контента в Open edX?"],
            ["Как настроить систему уведомлений в Open edX?"]
        ]
        
        gr.Examples(
            examples=examples,
            inputs=query_input,
            outputs=[answer_output, sources_output, token_stats_output],
            fn=lambda q: answer_query(q, DEFAULT_DB_PATH, 5, 0.2)
        )
        
        # Обработчики событий
        submit_btn.click(
            fn=answer_query,
            inputs=[query_input, db_path, k_slider, temperature_slider],
            outputs=[answer_output, sources_output, token_stats_output]
        )
        
        clear_btn.click(
            fn=lambda: ("", None, None),
            inputs=[],
            outputs=[answer_output, sources_output, token_stats_output]
        )
        
        reset_btn.click(
            fn=reset_token_counter,
            inputs=[],
            outputs=[gr.Textbox(label="Статус")]
        )
    
    return demo

if __name__ == "__main__":
    # Проверяем наличие директории с базой знаний
    if not os.path.exists(DEFAULT_DB_PATH):
        logger.warning(
            f"Директория с базой знаний не найдена: {DEFAULT_DB_PATH}. "
            "Создайте базу знаний перед запуском."
        )
    
    # Создаем и запускаем интерфейс
    demo = create_ui()
    demo.launch(share=True)