# Как создать RAG модель с нуля?

Оригинал статьи: https://practicaldev-herokuapp-com.freetls.fastly.net/hakeem/how-to-build-a-rag-model-from-scratch-bpf

Автор: Хаким Аббас<br />
Posted on 28 окт. 2024 г.

## Обзор
Модель `RAG` (Retrieval-Augmented Generation) - это гибридная система, сочетающая в себе два метода искусственного интеллекта:

* Модели на основе поиска: системы, которые ищут релевантную информацию в большом наборе документов или баз знаний.
* Генеративные модели: системы искусственного интеллекта, такие как GPT-3, которые могут генерировать связный текст по запросу. Идея RAG заключается в том, чтобы объединить сильные стороны обоих подходов. Вместо того чтобы генерировать ответы на основе фиксированной модели, которая ограничена в объёме знаний и может выдавать галлюцинации или выдумывать факты, RAG позволяет генеративному компоненту сначала извлекать реальные внешние знания, а затем генерировать ответы на основе этих извлечённых фактов. Например, при ответе на вопросы модель RAG будет:

  * Используйте ретривер для сбора соответствующих документов из базы данных.
  * Используйте генератор для синтеза окончательного ответа, объединив полученные документы с исходным запросом. Таким образом, модели RAG отлично справляются с задачами, требующими больших знаний, поскольку они могут получить доступ к большому и постоянно обновляемому хранилищу информации, а не полагаться исключительно на то, на чём обучалась языковая модель.

## Архитектура

Модель RAG состоит из двух основных архитектурных компонентов: ретривера (retriever) и генератора (generator). Каждый из них играет важную роль в обеспечении эффективности модели при создании точных ответов, основанных на знаниях.

### Retriever

Ретривер отвечает за поиск и извлечение соответствующих документов, отрывков или фрагментов из заранее созданной базы знаний. 

Обычно это включает в себя следующие шаги:
* Индексирование: перед выполнением запроса нам нужен предварительно созданный индекс документов или фрагментов знаний, созданный с помощью таких методов, как:
  * Поиск плотных проходов (DPR)
  * BM25 (классическая модель поиска на основе терминов)
  * Преобразователи предложений для поиска на основе встраивания
* Встраивание запроса: входной запрос сначала преобразуется в векторное представление с помощью модели встраивания, обычно это бикодер (например, ретривер на основе BERT). Затем этот вектор используется для поиска наиболее релевантных документов в индексе.
* Оценка документов: после получения векторных представлений поисковая система вычисляет показатель сходства между векторным представлением запроса и векторными представлениями документов. Затем выбираются первые k документов.

### Generator

Генератор отвечает за создание окончательного результата на основе входного запроса и полученных документов. Обычно этот компонент представляет собой предварительно обученную языковую модель, такую как BART или T5, которая настраивается для обработки полученных документов и создания ответов.
Ключевые элементы генератора:

* Форматирование ввода: полученные документы объединяются с запросом в единую последовательность ввода, которая передаётся языковой модели.
* Генерация текста: модель использует такие методы, как поиск по лучу или выборка ядер, для создания итогового результата на основе входной последовательности.

## Создание RAG модели

Создание модели RAG с нуля включает в себя несколько этапов: от создания ретривера до интеграции генератора. Ниже приведено пошаговое руководство по созданию простой модели RAG с использованием библиотеки Hugging Face Transformers и FAISS для эффективного поиска.

### Настройка среды

Для начала вам нужно будет установить необходимые зависимости:

<pre>pip install -r requirements.txt</pre>

или

<pre>
pip install transformers faiss-gpu sentence-transformers datasets
</pre>

Необходимые библиотеки:
* transformers - для предварительно обученных языковых моделей.
* faiss-gpu - для создания индекса документа и запроса к нему.
* sentence-transformers - для встраивания запросов и документов.
* datasets - для доступа к данным и их обработки.

### Подготовка данных

Для поиска вам понадобится большой корпус документов. Предположим, у вас есть набор документов, сохранённых в виде текстовых файлов. В идеале каждый документ должен быть небольшим (например, несколько абзацев), чтобы поиск был эффективным.
Мы будем использовать FAISS для индексирования и поиска по этим документам.

В примере используется этот текст:
<pre>
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load a pre-trained sentence transformer model for embedding
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

# Sample documents (replace with your actual corpus)
documents = [
    "The captial of France is Paris.",
    "The Eiffel Tower is a famous landmark is Paris.",
    "France is known for its wines and cuisine."
]

# Compute embeddings for the documents
document_embeddings = model.encode(documents, convert_to_tensor=False)

# Create a FAISS index and the document embeddings
dimension = document_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(document_embeddings))

# Save the document list for later retrieval
document_store = documents
</pre>

### Retrieval

Поисковый механизм использует индекс FAISS для поиска релевантных документов по заданному запросу. Вот как выполняется поиск:

<pre>
def retrieve_documents(query, top_k = 3):
    # Embed the query using the same model
    query_embedding = model.encode([query], convert_to_tensor=False)

    # Perform the search in the FAISS index
    distances, indices = faiss_index.search(np.array(query_embedding), top_k)

    # Retrieve the top_k documents based on indices
    retrieved_docs = [document_store[idx] for idx in indices[0]]

    return retrieved_docs
</pre>

Теперь retrieval готов к использованию. Давайте протестируем его с помощью запроса:

<pre>
query = "What is the capital of France?"
retrieved_docs = retrieve_documents(query)
print(retrieved_docs)
</pre>

### Generator

Генератор будет основан на предварительно обученной модели «sequence-to-sequence», такой как BART или T5, которая способна генерировать текст на основе запроса и дополнительного контекста (найденных документов).

Мы объединим найденные документы и запрос в единый ввод для генератора:

<pre>
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load a pre-trained T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-base')
generator_model = T5ForConditionalGeneration.from_pretrained('t5-base')

def generate_answer(query, retrieved_docs):
    # Combine the query and the retrieved documents into a single input sequence
    input_text = f"Query: {query} \nDocuments: {' '.join(retrieved_docs)}"

    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)

    # Generate an answer using the model
    outputs = generator_model.generate(inputs['input_ids'], max_length=150, num_beams=3, early_stopping=True)

    # Decode the output into text
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer
</pre>

### Собираем вместе

Наконец, мы объединяем Retrieval и Generator в единый конвейер, который принимает запрос и выводит сгенерированный ответ:

<pre>
def rag_pipeline(query):
    # Retrieve relevant documents
    retrieved_docs = retrieve_documents(query)

    # Generate the final answer using the generator
    answer = generate_answer(query, retrieved_docs)

    return answer
</pre>

<pre>
query = "What is France known for?"
answer = rag_pipeline(query)
print(answer)
</pre>

Вы увидите, как генератор выдаёт ответ на основе полученных документов, опираясь на реальные знания.

### Fine-Tuning ретривера

Тонкая настройка ретривера может значительно повысить его производительность. Модель с двумя кодировщиками (например, DPR) часто настраивается на наборе данных пар «вопрос-ответ» для оптимизации процесса поиска.

Чтобы выполнить тонкую настройку ретривера:
* Соберите набор данных, содержащий запросы и соответствующие им документы.
* Используйте контрастную потерю, чтобы обучить модель максимизировать сходство между запросами и релевантными документами и минимизировать сходство с нерелевантными документами.
* Обновите индекс FAISS с помощью встраиваний от точно настроенного ретривера.

### Fine-Tuning генератора

Аналогичным образом генератор можно дообучить на таких наборах данных, как SQuAD, TriviaQA или пользовательских парах «вопрос-ответ». Процесс дообучения включает в себя обучение модели генерированию связных ответов на основе запроса и найденных документов.

Ключевые шаги:
* Соберите (или создайте) набор данных, состоящий из запросов, документов и ответов.
* Выполните точную настройку модели «последовательность-последовательность» на этом наборе данных с помощью функции потерь кросс-энтропии.
* Проверяйте и корректируйте гиперпараметры, такие как скорость обучения и размер пакета.