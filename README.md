# SimpleRAG

Một hệ thống RAG (Retrieval-Augmented Generation) đơn giản viết bằng Python dùng để tìm kiếm và trả lời các câu hỏi về chủ đề **Machine Learning (Học máy)** bằng tiếng Việt.

Hệ thống cho phép người dùng hỏi đáp trực tiếp với nguồn tài liệu cục bộ dựa trên phương pháp tìm kiếm ngữ nghĩa (semantic search) và sinh câu trả lời bằng mô hình ngôn ngữ lớn (Gemini).

---

## 🛠️ Tính năng chính
- **Đa dạng mô hình nhúng (Embedding)**: Hỗ trợ nhúng tài liệu với nhiều tùy chọn:
  - `gemini-embedding-001` (qua Google GenAI SDK)
  - `multilingual-e5-base` (chạy offline qua Hugging Face `sentence-transformers`)
  - `multilingual-e5-large` (chạy offline qua Hugging Face `sentence-transformers`)
- **Bộ máy truy xuất (Retrieval)**: Sử dụng độ tương đồng Cosine (Cosine Similarity) để xếp hạng và tìm kiếm tài liệu tương quan nhất với câu hỏi.
- **Chatbot thông minh**: Sinh câu trả lời tối ưu hóa bằng mô hình `gemini-3.1-flash-lite`, đảm bảo câu trả lời bám sát nội dung tài liệu và không phát sinh thông tin giả (hallucination).
- **Bộ công cụ đánh giá (Evaluation)**:
  - Đánh giá giai đoạn truy xuất: Tính toán các chỉ số `Hit@K` (1, 3) và `Recall@K` (1, 3, 5).
  - Đánh giá giai đoạn sinh: Sử dụng LLM làm giám khảo (`gemini-3.1-flash-lite` hoặc `gemini-2.5-flash`) để đối chiếu câu trả lời tự động với nhãn chuẩn (Ground Truth).
  - Đánh giá khả năng chống ảo tưởng (Hallucination): Kiểm thử khả năng từ chối trả lời của mô hình khi gặp câu hỏi ngoài Context tài liệu.
  - Lưu trữ kết quả kiểm thử: Tự động kết xuất báo cáo kiểm thử ra các file log tương ứng trong thư mục `test_results/`.

---

## 📂 Cấu trúc thư mục dự án

```text
SimpleRAG/
├── docs/                     # Kho tài liệu học máy dạng text (.txt)
├── embeds/                   # Lưu trữ file vector nhúng (.npy) sau khi chạy
├── tests/
│   ├── evaluation.json       # Tập dữ liệu câu hỏi, nhãn chuẩn và tài liệu liên quan để đánh giá QA
│   └── hallucnation_test.json # Tập dữ liệu câu hỏi ngoài lề để đánh giá khả năng chống ảo tưởng
├── test_results/             # Thư mục lưu kết quả chạy kiểm thử (tự động tạo)
│   ├── test_retrieval_result.txt
│   ├── test_qa_result.txt
│   └── test_hallucination_result.txt
├── .env                      # File cấu hình khóa API
├── .gitignore
├── requirements.txt          # Các thư viện Python cần thiết
├── embedding.py              # Script sinh và lưu trữ vector nhúng cho tài liệu
├── retrieval.py              # Script tìm kiếm tài liệu tương tự từ câu hỏi
├── chatbot.py                # Giao diện dòng lệnh (CLI) hỏi đáp với Chatbot
├── test_retrieval.py         # Kiểm thử và đo lường chất lượng truy xuất
├── test_qa.py                # Đánh giá chất lượng trả lời tự động bằng LLM Judge
└── test_hallucination.py     # Đánh giá khả năng chống ảo tưởng của chatbot
```

---

## 🚀 Hướng dẫn cài đặt và sử dụng

### 1. Chuẩn bị môi trường và Cài đặt thư viện
Đảm bảo bạn đã cài đặt Python 3.10+. Chạy lệnh sau để cài đặt các thư viện phụ thuộc:

```bash
pip install -r requirements.txt
```

### 2. Cấu hình khóa API Gemini
Tạo tệp `.env` ở thư mục gốc của dự án với nội dung như sau:

```env
GEMINI=your_gemini_api_key_here
```
*(Thay `your_gemini_api_key_here` bằng API Key thực tế của bạn từ Google AI Studio).*

### 3. Quy trình chạy ứng dụng

#### Bước 1: Nhúng tài liệu (Embedding)
Trước khi hỏi đáp hoặc đánh giá, bạn cần chạy bước này để tạo và lưu trữ các vector nhúng của tài liệu trong thư mục `./docs/`:
```bash
python embedding.py
```
Nhập lựa chọn mô hình nhúng tương ứng khi được yêu cầu (`0` cho Gemini, `1` cho E5-base, `2` cho E5-large).

#### Bước 2: Hỏi đáp với Chatbot
Chạy chatbot để tương tác qua giao diện dòng lệnh:
```bash
python chatbot.py
```
Nhập câu hỏi của bạn. Chatbot sẽ tự động tìm kiếm tài liệu liên quan trong kho dữ liệu và sinh câu trả lời tương ứng. Nhập `exit` để thoát.

#### Bước 3: Đánh giá chất lượng tìm kiếm (Retrieval Evaluation)
Để đo lường hiệu quả tìm kiếm tài liệu của các mô hình nhúng khác nhau, chạy tệp kiểm thử:
```bash
python test_retrieval.py
```
Chương trình sẽ in ra các chỉ số `Mean hit@1`, `Mean hit@3`, `Mean recall@1`, `Mean recall@3`, `Mean recall@5` và danh sách các câu hỏi tìm kiếm sai. Kết quả cũng sẽ được lưu vào file `test_results/test_retrieval_result.txt`.

#### Bước 4: Đánh giá chất lượng trả lời (QA Evaluation)
Để kiểm tra xem Chatbot trả lời có đúng so với câu trả lời mẫu hay không bằng cách dùng LLM chấm điểm:
```bash
python test_qa.py
```
Kết quả chấm điểm (Đúng/Sai tương ứng `1`/`0`) sẽ được thống kê ở cuối tiến trình kèm các trường hợp trả lời sai để dễ dàng gỡ lỗi. Chi tiết kết quả sẽ được ghi vào file `test_results/test_qa_result.txt`.

#### Bước 5: Đánh giá chống ảo tưởng (Hallucination Evaluation)
Để đánh giá khả năng phản hồi "Tôi không tìm thấy thông tin trong tài liệu." khi chatbot gặp câu hỏi ngoài phạm vi tài liệu:
```bash
python test_hallucination.py
```
Màn hình sẽ hiển thị tiến độ chạy và kết xuất kết quả tổng quan (số test pass, danh sách top 5 test fail). Kết quả chi tiết sẽ được lưu tự động tại file `test_results/test_hallucination_result.txt`.
