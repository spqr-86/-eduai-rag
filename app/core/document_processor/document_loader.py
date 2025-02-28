from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import logging

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    CSVLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    DirectoryLoader
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Универсальный загрузчик документов, поддерживающий различные форматы."""
    
    LOADER_MAPPING = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".html": UnstructuredHTMLLoader,
        ".htm": UnstructuredHTMLLoader,
        ".csv": CSVLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".ppt": UnstructuredPowerPointLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
    }
    
    @classmethod
    def load_document(cls, file_path: Union[str, Path]) -> List[Document]:
        """
        Загружает документ на основе его расширения.
        
        Args:
            file_path: Путь к файлу для загрузки
            
        Returns:
            List[Document]: Список документов LangChain
            
        Raises:
            ValueError: Если формат файла не поддерживается
        """
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        if file_extension not in cls.LOADER_MAPPING:
            supported_formats = ", ".join(cls.LOADER_MAPPING.keys())
            raise ValueError(
                f"Неподдерживаемый формат файла: {file_extension}. "
                f"Поддерживаемые форматы: {supported_formats}"
            )
        
        loader_class = cls.LOADER_MAPPING[file_extension]
        try:
            loader = loader_class(str(file_path))
            documents = loader.load()
            logger.info(f"Загружено {len(documents)} фрагментов из {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Ошибка при загрузке {file_path}: {str(e)}")
            raise
    
    @classmethod
    def load_directory(cls, 
                       dir_path: Union[str, Path], 
                       glob_pattern: str = "**/*.*",
                       recursive: bool = True) -> List[Document]:
        """
        Загружает все поддерживаемые документы из указанной директории.
        
        Args:
            dir_path: Путь к директории
            glob_pattern: Шаблон для фильтрации файлов
            recursive: Рекурсивный поиск в поддиректориях
            
        Returns:
            List[Document]: Объединенный список документов
        """
        dir_path = Path(dir_path)
        
        # Создаем список поддерживаемых расширений для фильтрации
        supported_extensions = list(cls.LOADER_MAPPING.keys())
        
        all_documents = []
        
        for extension in supported_extensions:
            # Создаем шаблон для конкретного расширения
            extension_pattern = f"**/*{extension}" if recursive else f"*{extension}"
            
            try:
                # Используем DirectoryLoader для каждого типа файлов
                loader_class = cls.LOADER_MAPPING[extension]
                directory_loader = DirectoryLoader(
                    str(dir_path),
                    glob=extension_pattern,
                    loader_cls=loader_class
                )
                
                documents = directory_loader.load()
                logger.info(f"Загружено {len(documents)} документов с расширением {extension}")
                all_documents.extend(documents)
            except Exception as e:
                logger.warning(f"Ошибка при загрузке файлов {extension}: {str(e)}")
        
        return all_documents