import json
import os
from google import genai
from retrieval import retrieval
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

print("Choose your embedding model:")
print("gemini-embedding-001 : press 0")
print("multilingual-e5-base : press 1")
print("multilingual-e5-large : press 2")
choice = input("Enter your choice: ")

acc = 0
false_retrieval = []

with open("./tests\evaluation.json", "r", encoding="utf-8") as file:
    test = json.load(file)
print("Load test question successfully.")

if choice == "0":
    load_dotenv()
    api_key = os.getenv("GEMINI_EMBEDDING_1")
    client = genai.Client(api_key=api_key)

    print("Retrieving...")
    for i, block in enumerate(test):
        question = block["question"]
        target_relevant_docs = set(block["relevant_documents"])
        result = client.models.embed_content(
            model = "gemini-embedding-001",
            contents = question
        )

        input_vector = result.embeddings[0].values
        retrieval_result = retrieval(input_vector, ".\embeds\gemini_embedding.npy")
        predict_relevant_docs = set([t[1] for t in retrieval_result])
        
        if target_relevant_docs.intersection(predict_relevant_docs) == target_relevant_docs:
            acc += 1
        else:
            data = {
                "question" : question,
                "predict_relevant" : list(predict_relevant_docs),
                "actual_relevant" : list(target_relevant_docs)
            }
            false_retrieval.append(data)
    print("Done")

    acc = acc * 100 / len(test)
    print(f"Accuracy: {acc:.4f}%")
    print(f"Top 5 false retrival:")
    for i in range(len(false_retrieval)):
        print(false_retrieval[i])
        print()
        if i >= 4:
            break

elif choice == "1":
    client = SentenceTransformer("intfloat/multilingual-e5-base")

    print("Retrieving...")
    for i, block in enumerate(test):
        question = block["question"]
        target_relevant_docs = set(block["relevant_documents"])
        input_vector = client.encode(f"querry: {question}")
        retrieval_result = retrieval(input_vector, ".\embeds\mul_e5_base_embedding.npy")
        predict_relevant_docs = set([t[1] for t in retrieval_result])
        
        if target_relevant_docs.intersection(predict_relevant_docs) == target_relevant_docs:
            acc += 1
        else:
            data = {
                "question" : question,
                "predict_relevant" : list(predict_relevant_docs),
                "actual_relevant" : list(target_relevant_docs)
            }
            false_retrieval.append(data)
    print("Done")

    acc = acc * 100 / len(test)
    print(f"Accuracy: {acc:.4f}%")
    print(f"Top 5 false retrival:")
    for i in range(len(false_retrieval)):
        print(false_retrieval[i])
        print()
        if i >= 4:
            break

elif choice == "2":
    client = SentenceTransformer("intfloat/multilingual-e5-large")

    print("Retrieving...")
    for i, block in enumerate(test):
        question = block["question"]
        target_relevant_docs = set(block["relevant_documents"])
        input_vector = client.encode(f"querry: {question}")
        retrieval_result = retrieval(input_vector, ".\embeds\mul_e5_large_embedding.npy")
        predict_relevant_docs = set([t[1] for t in retrieval_result])
        
        if target_relevant_docs.intersection(predict_relevant_docs) == target_relevant_docs:
            acc += 1
        else:
            data = {
                "question" : question,
                "predict_relevant" : list(predict_relevant_docs),
                "actual_relevant" : list(target_relevant_docs)
            }
            false_retrieval.append(data)
    print("Done")

    acc = acc * 100 / len(test)
    print(f"Accuracy: {acc:.4f}%")
    print(f"Top 5 false retrival:")
    for i in range(len(false_retrieval)):
        print(false_retrieval[i])
        print()
        if i >= 4:
            break