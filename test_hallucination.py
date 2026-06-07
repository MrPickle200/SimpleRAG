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
    embed = SentenceTransformer("intfloat/multilingual-e5-base")
    return generator, embed

def load_test():
    # Attempt both tests/hallucnation_test.json and tests/hallucination_test.json
    possible_paths = ["tests/hallucnation_test.json", "tests/hallucination_test.json", "tests\\hallucnation_test.json"]
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "r", encoding = "utf-8") as file:
                test = json.load(file)
            return test
    raise FileNotFoundError("Could not find hallucnation_test.json or hallucination_test.json")

def main():
    generator, embed = init()
    print("Init model successfully.")
    test = load_test()
    print("Load test successfully.")

    failed_test = []
    passed_count = 0

    print("Testing hallucination...")
    for i, block in enumerate(test):
        print(f"\r[{'*' * (i + 1)}{'-' * (len(test) - i - 1)}]", end = "")
        question = block["question"]
        prompt = get_prompt(question, embed)
        
        try:
            response = generator.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    thinking_config=genai.types.ThinkingConfig(thinking_budget=0)
                )
            )
        except Exception:
            response = generator.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    thinking_config=genai.types.ThinkingConfig(thinking_budget=0)
                )
            )

        prediction = response.text.strip()
        
        # Check if the answer is the expected refusion or not
        # If it is NOT "Tôi không tìm thấy thông tin trong tài liệu.", it hallucinated.
        if prediction != "Tôi không tìm thấy thông tin trong tài liệu.":
            block["prediction"] = prediction
            failed_test.append(block)
        else:
            passed_count += 1
            
        time.sleep(5)
        
    print()
    print("Done.")
    print()

    output_lines = []
    output_lines.append(f"Total passed test: {passed_count}/{len(test)}")
    output_lines.append("")

    if failed_test:
        output_lines.append("Top 5 failed test (hallucinated)")
        for i, block in enumerate(failed_test):
            output_lines.append(str(block))
            output_lines.append("")

            if i >= 4:
                break

    output_content = "\n".join(output_lines)
    print(output_content)

    os.makedirs("test_results", exist_ok=True)
    with open("test_results/test_hallucination_result.txt", "w", encoding="utf-8") as out_file:
        out_file.write(output_content)

if __name__ == "__main__":
    main()
