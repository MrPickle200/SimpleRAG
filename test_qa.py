from chatbot import get_prompt
from google import genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import json
import os
import time

load_dotenv()

def init():
    API_KEY = os.getenv("GEMINI")
    generator = genai.Client(api_key = API_KEY)
    judge = genai.Client(api_key = API_KEY)
    embed = SentenceTransformer("intfloat/multilingual-e5-base")
    return generator, judge, embed

def load_test():
    with open("tests\evaluation.json", "r", encoding = "utf-8") as file:
        test = json.load(file)
    return test

def get_score(model, question: str, ground_truth : str, prediction : str):
    prompt = f"""
        Bạn là một kỹ sư Machine Learning với 10 năm kinh nghiệm.
        Nhiệm vụ của bạn là đánh giá câu trả lời (prediction) cho câu hỏi (question)
        dựa trên Ground truth

        Question:
        {question}

        Ground Truth:
        {ground_truth}
        
        Prediction:
        {prediction}
        
        Đánh giá:

        0 = sai
        1 = đúng

        Chỉ trả về 0 hoặc 1.
    """

    try:
        response = model.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                thinking_config=genai.types.ThinkingConfig(thinking_budget=0)
            )
        )
    except:
        response = model.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                thinking_config=genai.types.ThinkingConfig(thinking_budget=0)
            )
        )

    score = response.text.strip()
    try:
        return int(score)
    except:
        print(f"Invalid format for score: {score}")
        return 0

def main():
    generator, judge, embed = init()
    print("Init model successfully.")
    test = load_test()
    print("Load test successfully.")

    failed_test = []
    total_score = 0

    print("Testing...")
    for i, block in enumerate(test):
        print(f"\r[{'*' * (i + 1)}{'-' * (len(test) - i - 1)}]", end = "")
        question = block["question"]
        ground_truth = block["ground_truth_answer"]
        prompt = get_prompt(question, embed)
        try:
            response = generator.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    thinking_config=genai.types.ThinkingConfig(thinking_budget=0)
                )
            )
        except:
            response = generator.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    thinking_config=genai.types.ThinkingConfig(thinking_budget=0)
                )
            )

        prediction = response.text.strip()
        time.sleep(5)
        score = get_score(judge, question, ground_truth, prediction)
        total_score += score

        # print(f"[Q]: {question}")
        # print(f"[A]: {prediction}")
        # print(f"[Score]: {score}")
        # print()

        if score == 0:
            block["prediction"] = prediction
            failed_test.append(block)
    print()
    print("Done.")
    print()
    print(f"Total correct test: {total_score}/{len(test)}")
    print()

    if failed_test:
        print("Top 5 failed test")
        for i, block in enumerate(failed_test):
            print(block)
            print()

            if i >= 4:
                break

if __name__ == "__main__":
    main()      