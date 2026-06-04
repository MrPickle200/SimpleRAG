from google import genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np
import json
import os

load_dotenv()

print("Choose your embedding model:")
print("gemini-embedding-001 : press 0")
print("multilingual-e5-base : press 1")
print("multilingual-e5-large : press 2")
choice = input("Enter your choice: ")

if choice == "0":
    API_KEY = os.getenv("GEMINI_EMBEDDING_1")
    client = genai.Client(api_key = API_KEY)
    print("Load model successfully.")
    len_docs = len(os.listdir("./docs")) - 1
    docs = []

    print("Start embeddding...")
    for i, file in enumerate(os.listdir("./docs")):
        # Progress bar
        progress_bar = "*" * i + "-" * (len_docs - i)
        print(f"\r[{progress_bar}]", end = "")
        
        # Embedding
        file_name = file.split(".")[0]
        file_path = os.path.join("./docs", file)
        with open(file_path, "r", encoding="utf-8") as content:
            join_content = " ".join(content)
        
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=join_content
        )
        embedding = result.embeddings[0].values        
        tmp = {
            "id" : i,
            "doc" : file,
            "embedding" : embedding,
            "text" : join_content
        }
        docs.append(tmp)
    print()
    print("Done.")

    np.save("./embeds/gemini_embedding", docs)

    # with open("./embeds/gemini_embedding.json", "w", encoding="utf-8") as file:
    #     json.dump(docs, file, ensure_ascii=False, indent=4)
    #     print("Save sucessfully.")

elif choice == "1":
    model = SentenceTransformer("intfloat/multilingual-e5-base")
    print("Load model successfully.")
    len_docs = len(os.listdir("./docs")) - 1
    docs = []

    print("Start embeddding...")
    for i, file in enumerate(os.listdir("./docs")):
        # Progress bar
        progress_bar = "*" * i + "-" * (len_docs - i)
        print(f"\r[{progress_bar}]", end = "")
        
        # Embedding
        file_name = file.split(".")[0]
        file_path = os.path.join("./docs", file)
        with open(file_path, "r", encoding="utf-8") as content:
            join_content = " ".join(content)
        
        embedding = model.encode(f"passage: {join_content}").tolist()
        tmp = {
            "id" : i,
            "doc" : file,
            "embedding" : embedding,
            "text" : join_content
        }
        docs.append(tmp)
    print()
    print("Done.")

    np.save("./embeds/mul_e5_base_embedding", docs)

    # with open("./embeds/mul_e5_base_embedding.json", "w", encoding="utf-8") as file:
    #     json.dump(docs, file, ensure_ascii=False, indent=4)
    #     print("Save sucessfully.")

elif choice == "2":
    model = SentenceTransformer("intfloat/multilingual-e5-large")
    print("Load model successfully.")
    len_docs = len(os.listdir("./docs")) - 1
    docs = []

    print("Start embeddding...")
    for i, file in enumerate(os.listdir("./docs")):
        # Progress bar
        progress_bar = "*" * i + "-" * (len_docs - i)
        print(f"\r[{progress_bar}]", end = "")
        
        # Embedding
        file_name = file.split(".")[0]
        file_path = os.path.join("./docs", file)
        with open(file_path, "r", encoding="utf-8") as content:
            join_content = " ".join(content)
        
        embedding = model.encode(f"passage: {join_content}").tolist()
        tmp = {
            "id" : i,
            "doc" : file,
            "embedding" : embedding,
            "text" : join_content
        }
        docs.append(tmp)
    print()
    print("Done.")

    np.save("./embeds/mul_e5_large_embedding", docs)
    
    # with open("./embeds/mul_e5_base_embedding.json", "w", encoding="utf-8") as file:
    #     json.dump(docs, file, ensure_ascii=False, indent=4)
    #     print("Save sucessfully.")

else:
    print("Invalid choice.")

    