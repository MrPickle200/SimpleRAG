import os
import json
import time
import numpy as np
from google import genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

def cosine_similarity(vector_1, vector_2):
    return np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))

def calculate_similarity(input_vector, pth_to_embed):
    contents = np.load(pth_to_embed, allow_pickle = True)
    similarity_result = []
    for block in contents:
        embed_vector = block["embedding"]
        doc = block["doc"]
        idx = block["id"]

        sim = cosine_similarity(input_vector, embed_vector)
        similarity_result.append((idx, doc, sim))
    
    return sorted(similarity_result, key = lambda x: x[-1], reverse = True)

def get_top_k(similarity_result, top_k = 5):
    return similarity_result[ : top_k]

def retrieval(input_vector, pth_to_embed):
    sim_result = calculate_similarity(input_vector, pth_to_embed)
    top_k = get_top_k(sim_result)
    return top_k

def main():
    print("Choose your embedding model:")
    print("gemini-embedding-001 : press 0")
    print("multilingual-e5-base : press 1")
    print("multilingual-e5-large : press 2")
    choice = input("Enter your choice: ")

    if choice == "0":
        api_key = os.getenv("GEMINI_EMBEDDING_1")
        user_input = input("Enter your question: ")
        
        print("Embedding your question...")
        client = genai.Client(api_key=api_key)
        start = time.time()
        result = client.models.embed_content(
            model = "gemini-embedding-001",
            contents = user_input
        )
        end = time.time()
        input_vector = result.embeddings[0].values
        print(f"Total time: {(end - start):.4f}s")
        print()

        print("Retrievialing...")
        start = time.time()
        res = retrieval(input_vector, ".\embeds\gemini_embedding.npy")
        end = time.time()
        print(f"Total time: {(end - start):.4f}s")
        print()

        print("=========Result========")
        print(f"Your question: {user_input}")
        print("Relevant docs:")
        for i, block in enumerate(res, start = 1):
            idx, doc, sim = block
            print(f"Top {i}")
            print(f"-Document: {doc}")
            print(f"-Score: {sim}")
            print()
    
    elif choice == "1":
        user_input = input("Enter your question: ")
        
        print("Embedding your question...")
        client = SentenceTransformer("intfloat/multilingual-e5-base")
        start = time.time()
        input_vector = client.encode(f"querry: {user_input}")
        end = time.time()
        print(f"Total time: {(end - start):.4f}s")
        print()

        print("Retrievialing...")
        start = time.time()
        res = retrieval(input_vector, "./embeds/mul_e5_base_embedding.npy")
        end = time.time()
        print(f"Total time: {(end - start):.4f}s")
        print()

        print("=========Result========")
        print(f"Your question: {user_input}")
        print("Relevant docs:")
        for i, block in enumerate(res, start = 1):
            idx, doc, sim = block
            print(f"Top {i}")
            print(f"-Document: {doc}")
            print(f"-Score: {sim}")
            print()

    elif choice == "2":
        user_input = input("Enter your question: ")
        
        print("Embedding your question...")
        client = SentenceTransformer("intfloat/multilingual-e5-large")
        start = time.time()
        input_vector = client.encode(f"querry: {user_input}")
        end = time.time()
        print(f"Total time: {(end - start):.4f}s")
        print()

        print("Retrievialing...")
        start = time.time()
        res = retrieval(input_vector, "./embeds/mul_e5_large_embedding.npy")
        end = time.time()
        print(f"Total time: {(end - start):.4f}s")
        print()

        print("=========Result========")
        print(f"Your question: {user_input}")
        print("Relevant docs:")
        for i, block in enumerate(res, start = 1):
            idx, doc, sim = block
            print(f"Top {i}")
            print(f"-Document: {doc}")
            print(f"-Score: {sim}")
            print()

    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()