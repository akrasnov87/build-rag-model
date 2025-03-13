from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForMaskedLM
import faiss
import numpy as np

model_id = "sentence-transformers/use-cmlm-multilingual"

# Load a pre-trained sentence transformer model for embedding
model = SentenceTransformer(model_id)
#model = AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True)

# Sample documents (replace with your actual corpus)
documents = [
    "Ведение артефактов проекта (Эпики, задачи)",
    "На портале публикуется подробная информация для разработчиков БД Postgres. (Архитектурные решения, стандарты, правила, лучшые практики и т.п.)",
    "На портале публикуется подробная информация по платформе iFlow. (Архитектура, конвенция, средства разработки и т.п.)",
    "Руководство системного администратора OmniUS 5.0 и платформы iFlow (развертывание, установка приложения и т.д.)",
    "Разработка бизнес-логики выполняется путем создания новых и изменения существующих объектов БД PosgreSQL."
]

# Compute embeddings for the documents
document_embeddings = model.encode(documents, convert_to_tensor=False)

# Create a FAISS index and the document embeddings
dimension = document_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(document_embeddings))

# Save the document list for later retrieval
document_store = documents

def retrieve_documents(query, top_k = 3):
    # Embed the query using the same model
    query_embedding = model.encode([query], convert_to_tensor=False)

    # Perform the search in the FAISS index
    distances, indices = faiss_index.search(np.array(query_embedding), top_k)

    # Retrieve the top_k documents based on indices
    retrieved_docs = [document_store[idx] for idx in indices[0]]

    return retrieved_docs

#query = "Что публикуется на портале?"
#retrieved_docs = retrieve_documents(query)
#print(retrieved_docs)

from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load a pre-trained T5 model and tokenizer
MODEL_NAME = 'cointegrated/rut5-base-paraphraser'
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
generator_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

def generate_answer(query, retrieved_docs):
    # Combine the query and the retrieved documents into a single input sequence
    input_text = f"Query: {query} \nDocuments: {' '.join(retrieved_docs)}"

    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)

    # Generate an answer using the model
    outputs = generator_model.generate(inputs['input_ids'], max_length=128, num_beams=3, early_stopping=True)

    # Decode the output into text
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer

def rag_pipeline(query):
    # Retrieve relevant documents
    retrieved_docs = retrieve_documents(query)

    # Generate the final answer using the generator
    answer = generate_answer(query, retrieved_docs)

    return answer

query = "Что ты знешь об iFlow?"
answer = rag_pipeline(query)
print(answer)