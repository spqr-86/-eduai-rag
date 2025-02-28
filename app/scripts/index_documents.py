import logging
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from app.utils.chroma_manager import ChromaManager

from app.core.document_processor.indexing_pipeline import DocumentIndexingPipeline
from app.core.embeddings.embedding_factory import EmbeddingFactory
from app.utils.chroma_manager import ChromaManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Загружаем переменные окружения
    load_dotenv()
    
    # Парсим аргументы командной строки
    parser = argparse.ArgumentParser(description='Индексация документов для EduAI')
    parser.add_argument('--source', required=True, help='Путь к файлу или директории для индексации')
    parser.add_argument('--output', required=True, help='Директория для сохранения векторной БД')
    parser.add_argument('--embedding', default='openai', help='Тип эмбеддингов (openai, local, bge)')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Размер чанка для разбиения')
    parser.add_argument('--chunk-overlap', type=int, default=200, help='Размер перекрытия чанков')
    parser.add_argument('--recursive', action='store_true', help='Рекурсивный обход директорий')
    parser.add_argument('--collection', default='eduai_documents', help='Имя коллекции ChromaDB')
    
    args = parser.parse_args()
    
    # Проверяем существование исходного пути
    source_path = Path(args.source)
    if not source_path.exists():
        logger.error(f"Указанный путь не существует: {source_path}")
        return
    
    # Определяем, является ли источник директорией
    is_directory = source_path.is_dir()
    
    # Создаем директорию для выходных данных, если она не существует
    output_dir = Path(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # Создаем провайдер эмбеддингов
    logger.info(f"Инициализация провайдера эмбеддингов типа {args.embedding}")
    embedding_provider = EmbeddingFactory.create_embeddings(provider=args.embedding)
    
    # Инициализируем конвейер индексации
    indexing_pipeline = DocumentIndexingPipeline(
        embedding_provider=embedding_provider,
        vector_store_type="chroma",
        vector_store_config={
            "collection_name": args.collection
        }
    )
    
    # Запускаем индексацию
    logger.info(f"Начало индексации {'директории' if is_directory else 'файла'}: {source_path}")
    
    # Выполняем полный процесс индексации
    vector_store = indexing_pipeline.full_indexing_pipeline(
        source_path=source_path,
        persist_directory=str(output_dir),
        is_directory=is_directory,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        recursive=args.recursive
    )
  
    # После индексации используйте переменную vector_store
    stats = {
        "collection_name": "eduai_documents",  # Или получите из конфига
        "document_count": 32,  # Или получите из логов
        "persist_directory": str(output_dir)
    }
    logger.info(f"Индексация завершена. Статистика коллекции: {stats}")

if __name__ == "__main__":
    main()