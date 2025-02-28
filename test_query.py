# test_query.py
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate

# Инициализация компонентов
embeddings = OpenAIEmbeddings()
db = Chroma(
    persist_directory="data/chroma_db", 
    embedding_function=embeddings,
    collection_name="eduai_documents"
)

# Проверка количества документов
print(f"В базе {db._collection.count()} документов")

# Тестовый поиск
results = db.similarity_search("Что такое наследование в ООП?", k=2)
print("\nРезультаты поиска:")
for i, doc in enumerate(results):
    print(f"\nРезультат {i+1}:")
    print(f"Содержание: {doc.page_content[:200]}...")
    print(f"Метаданные: {doc.metadata}")

# Проверка RAG-цепочки
llm = ChatOpenAI(temperature=0.2)
prompt_template = ChatPromptTemplate.from_template("""
Ты - помощник для обучения объектно-ориентированному программированию. 
Отвечай на вопросы, используя только предоставленный контекст.

Контекст:
{context}

Вопрос: {question}
""")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": prompt_template}
)

print("\n\nТест RAG-цепочки:")
result = qa_chain.invoke({"query": "Объясни принципы ООП"})
print(result["result"])