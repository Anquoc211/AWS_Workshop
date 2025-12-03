---
title: "Ngày 36 - Giới thiệu NLP & Tiền xử lý văn bản"
weight: 1
chapter: false
pre: "<b> 1.8.1. </b>"
---

**Ngày:** 2025-10-27
**Trạng thái:** "Hoàn thành"  

---

# **Ghi chú bài giảng**

## Giới thiệu về Xử lý Ngôn ngữ Tự nhiên

### NLP là gì?

- NLP cho phép máy tính hiểu, diễn giải và tạo ra ngôn ngữ con người.
- Ứng dụng: chatbots, phân tích cảm xúc, dịch thuật, trợ lý giọng nói, công cụ tìm kiếm.
- Thách thức chính: tính mơ hồ, ngữ cảnh, thành ngữ, nhiều ngôn ngữ.

### Tổng quan quy trình NLP

```
Văn bản thô → Tiền xử lý → Trích xuất đặc trưng → Mô hình → Đầu ra
```

- Mỗi giai đoạn chuyển đổi văn bản thành dữ liệu có cấu trúc cho machine learning.
- Tiền xử lý rất quan trọng - nguyên tắc "rác vào, rác ra" được áp dụng.

## Kiến thức cơ bản về tiền xử lý văn bản

### Tokenization

- Chia văn bản thành các từ hoặc câu riêng lẻ.
- Word tokenization: "Hello world!" → ["Hello", "world", "!"]
- Sentence tokenization: Chia đoạn văn thành các câu.
- Tầm quan trọng: nền tảng cho tất cả các tác vụ NLP.

### Chuyển sang chữ thường

- Chuyển đổi tất cả văn bản sang chữ thường để đảm bảo tính nhất quán.
- "Hello" và "hello" được coi là cùng một từ.
- Đánh đổi: có thể mất thông tin (ví dụ: tên riêng).

### Loại bỏ dấu câu & ký tự đặc biệt

- Làm sạch văn bản bằng cách loại bỏ các ký tự không phải chữ và số.
- Chỉ giữ lại các từ có ý nghĩa để phân tích.
- Phụ thuộc vào ngữ cảnh: một số dấu câu có thể mang ý nghĩa.

## Những hiểu biết quan trọng

- NLP là cầu nối giữa giao tiếp của con người và sự hiểu biết của máy.
- Chất lượng tiền xử lý ảnh hưởng trực tiếp đến hiệu suất mô hình.
- Các tác vụ khác nhau yêu cầu các chiến lược tiền xử lý khác nhau.
- Luôn kiểm tra dữ liệu của bạn trước và sau khi tiền xử lý.

---

# **Thực hành Lab**

## Lab 1: Tokenization cơ bản

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Tải dữ liệu cần thiết
nltk.download('punkt')

text = "Natural Language Processing is fascinating! It enables many applications."

# Word tokenization
words = word_tokenize(text)
print("Words:", words)

# Sentence tokenization
sentences = sent_tokenize(text)
print("Sentences:", sentences)
```

## Lab 2: Làm sạch văn bản

```python
import re
import string

def clean_text(text):
    # Chữ thường
    text = text.lower()
    
    # Loại bỏ dấu câu
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Loại bỏ số
    text = re.sub(r'\d+', '', text)
    
    # Loại bỏ khoảng trắng thừa
    text = ' '.join(text.split())
    
    return text

sample = "Hello World! This is NLP 101."
cleaned = clean_text(sample)
print("Gốc:", sample)
print("Đã làm sạch:", cleaned)
```

## Lab 3: Khám phá NLTK

```python
import nltk

# Khám phá khả năng NLTK
print("Phiên bản NLTK:", nltk.__version__)

# Tải tài nguyên bổ sung
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords

# Xem stop words tiếng Anh
stop_words = set(stopwords.words('english'))
print(f"Số lượng stop words: {len(stop_words)}")
print("Mẫu stop words:", list(stop_words)[:10])
```

---

# **Bài tập thực hành**

1. Tokenize một đoạn văn từ cuốn sách yêu thích của bạn
2. Tạo một hàm loại bỏ URLs khỏi văn bản
3. So sánh số lượng từ trước và sau khi tiền xử lý
4. Thử nghiệm với các phương pháp tokenization khác nhau
