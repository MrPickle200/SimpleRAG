from google import genai
from dotenv import load_dotenv
from retrieval import retrieval
from sentence_transformers import SentenceTransformer
import os
import time

load_dotenv()

def generate_prompt(question : str, retrieval_result : list):
    context = ""
    for block in retrieval_result:
        file_name = block[1]
        path = os.path.join("docs", file_name)
        with open(path, "r", encoding="utf-8") as f:
            context += f"\n[Documents: {file_name}]"
            for line in f:
                context += f"\n{line}"

    prompt = f"""
        Bạn là một trợ lý hỏi đáp về Machine Learning.
        Bạn sẽ được cung cấp một tập các đoạn tài liệu đã được truy xuất từ hệ thống RAG.
        Yêu cầu:
        * Trả lời câu hỏi chỉ dựa trên các đoạn tài liệu được cung cấp.
        * Nếu nhiều đoạn tài liệu chứa thông tin liên quan, hãy tổng hợp chúng thành một câu trả lời thống nhất.
        * Không sử dụng kiến thức bên ngoài Context.
        * Trả lời thuần text, không dùng kí hiệu LaTex
        * Nếu Context không đủ để trả lời, hãy trả lời chính xác:
        "Tôi không tìm thấy thông tin trong tài liệu."
        Context:
        {context}
        Question:
        {question}
        Answer:
    """

    return prompt

def encode_question(model, question : str):
    return model.encode(f"querry: {question}")

def get_prompt(question: str, model):
    input_embed = encode_question(model, question)
    retrieval_result = retrieval(input_embed, "embeds\mul_e5_base_embedding.npy")
    prompt = generate_prompt(question, retrieval_result)
    return prompt        

def main():
    API_KEY = os.getenv("GEMINI")
    client = genai.Client(api_key = API_KEY)
    model = SentenceTransformer("intfloat/multilingual-e5-base")

    print("Xin chào, tôi là chatbot về Machine Learning của bạn. Let's start.")
    while True:
        question = input("[Q]: ")
        if "exit" in question:
            print("[A]: Bye.")
            break

        start = time.time()
        prompt = get_prompt(question, model)
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                thinking_config=genai.types.ThinkingConfig(thinking_budget=0)
                )
            )
        end = time.time()

        text = response.text.strip()
        print(f"[A]: {text}")
        print(f"[time]: {(end - start):.2f}s")


if __name__ == "__main__":
    main()
