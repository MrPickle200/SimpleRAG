import json
import os
from google import genai
from retrieval import retrieval
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

def recall_k(predict : list, target : list, k = 1):
    sub_predict = predict[ : k]
    return len(set(target).intersection(set(sub_predict))) / len(set(target))

def hit_k(predict : list, target : list, k = 1):
    sub_predict = predict[ : k]
    return int(len(set(target).intersection(set(sub_predict))) > 0)

print("Choose your embedding model:")
print("gemini-embedding-001 : press 0")
print("multilingual-e5-base : press 1")
print("multilingual-e5-large : press 2")
choice = input("Enter your choice: ")

hit_1 = 0
hit_3 = 0
recall_1 = 0
recall_3 = 0
recall_5 = 0
false_retrieval = []

with open("./tests\evaluation.json", "r", encoding="utf-8") as file:
    test = json.load(file)
print("Load test question successfully.")

if choice == "0":
    model_name = "gemini-embedding-001"
    load_dotenv()
    api_key = os.getenv("GEMINI")
    client = genai.Client(api_key=api_key)

    print("Retrieving...")
    for i, block in enumerate(test):
        question = block["question"]
        target_relevant_docs = block["relevant_documents"]
        result = client.models.embed_content(
            model = "gemini-embedding-001",
            contents = question
        )

        input_vector = result.embeddings[0].values
        retrieval_result = retrieval(input_vector, ".\embeds\gemini_embedding.npy")
        predict_relevant_docs = [t[1] for t in retrieval_result]
        
        hit_1 += hit_k(predict_relevant_docs, target_relevant_docs, k = 1)
        hit_3 += hit_k(predict_relevant_docs, target_relevant_docs, k = 3)
        
        recall_1 += recall_k(predict_relevant_docs, target_relevant_docs, k = 1)
        recall_3 += recall_k(predict_relevant_docs, target_relevant_docs, k = 3)
        recall_5 += recall_k(predict_relevant_docs, target_relevant_docs, k = 5)

        if recall_5 < 0.5:
            data = {
                "question" : question,
                "predict_relevant" : list(predict_relevant_docs),
                "actual_relevant" : list(target_relevant_docs)
            }
            false_retrieval.append(data)

elif choice == "1":
    model_name = "multilingual-e5-base"
    client = SentenceTransformer("intfloat/multilingual-e5-base")

    print("Retrieving...")
    for i, block in enumerate(test):
        question = block["question"]
        target_relevant_docs = block["relevant_documents"]
        input_vector = client.encode(f"querry: {question}")
        retrieval_result = retrieval(input_vector, ".\embeds\mul_e5_base_embedding.npy")
        predict_relevant_docs = [t[1] for t in retrieval_result]
        
        hit_1 += hit_k(predict_relevant_docs, target_relevant_docs, k = 1)
        hit_3 += hit_k(predict_relevant_docs, target_relevant_docs, k = 3)
                
        recall_1 += recall_k(predict_relevant_docs, target_relevant_docs, k = 1)
        recall_3 += recall_k(predict_relevant_docs, target_relevant_docs, k = 3)
        recall_5 += recall_k(predict_relevant_docs, target_relevant_docs, k = 5)

        if recall_5 < 0.5:
            data = {
                "question" : question,
                "predict_relevant" : list(predict_relevant_docs),
                "actual_relevant" : list(target_relevant_docs)
            }
            false_retrieval.append(data)

elif choice == "2":
    model_name = "multilingual-e5-large"
    client = SentenceTransformer("intfloat/multilingual-e5-large")

    print("Retrieving...")
    for i, block in enumerate(test):
        question = block["question"]
        target_relevant_docs = block["relevant_documents"]
        input_vector = client.encode(f"querry: {question}")
        retrieval_result = retrieval(input_vector, ".\embeds\mul_e5_large_embedding.npy")
        predict_relevant_docs = [t[1] for t in retrieval_result]
        
        hit_1 += hit_k(predict_relevant_docs, target_relevant_docs, k = 1)
        hit_3 += hit_k(predict_relevant_docs, target_relevant_docs, k = 3)
        
        recall_1 += recall_k(predict_relevant_docs, target_relevant_docs, k = 1)
        recall_3 += recall_k(predict_relevant_docs, target_relevant_docs, k = 3)
        recall_5 += recall_k(predict_relevant_docs, target_relevant_docs, k = 5)

        if recall_5 < 0.5:
            data = {
                "question" : question,
                "predict_relevant" : list(predict_relevant_docs),
                "actual_relevant" : list(target_relevant_docs)
            }
            false_retrieval.append(data)
    

print("Done")

output_lines = []
output_lines.append(f"Model: {model_name}")
output_lines.append(f"Mean hit@1: {(hit_1 / len(test)):.4f}")
output_lines.append(f"Mean hit@3: {(hit_3 / len(test)):.4f}")
output_lines.append(f"Mean recall@1: {(recall_1 / len(test)):.4f}")
output_lines.append(f"Mean recall@3: {(recall_3 / len(test)):.4f}")
output_lines.append(f"Mean recall@5: {(recall_5 / len(test)):.4f}")
if len(false_retrieval) >= 5:
    output_lines.append(f"Top 5 false retrival:")
    for i in range(len(false_retrieval)):
        output_lines.append(str(false_retrieval[i]))
        output_lines.append("")
        if i >= 4:
            break

output_content = "\n".join(output_lines)
print(output_content)

os.makedirs("test_results", exist_ok=True)
with open("test_results/test_retrieval_result.txt", "w", encoding="utf-8") as out_file:
    out_file.write(output_content)
