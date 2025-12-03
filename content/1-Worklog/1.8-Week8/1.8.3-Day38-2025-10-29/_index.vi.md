---
title: "Ngày 38 - Phân tích văn bản & Phân tích cảm xúc"
weight: 3
chapter: false
pre: "<b> 1.8.3. </b>"
---

**Ngày:** 2025-10-29
**Trạng thái:** "Hoàn thành"  

---

# **Ghi chú bài giảng**

## Phân tích văn bản cơ bản

### Phân tích tần suất từ

- Đếm tần suất xuất hiện của mỗi từ trong văn bản.
- Xác định các thuật ngữ và chủ đề phổ biến nhất.
- Nền tảng cho nhiều ứng dụng NLP.
- Hữu ích để hiểu nội dung tài liệu một cách nhanh chóng.

### N-grams

- Chuỗi N từ liên tiếp.
- Unigrams: từ đơn ("machine")
- Bigrams: hai từ ("machine learning")
- Trigrams: ba từ ("natural language processing")
- Nắm bắt ngữ cảnh và cụm từ phổ biến.

### Thống kê văn bản

- Độ dài tài liệu (từ, câu, ký tự)
- Kích thước từ vựng (từ duy nhất)
- Độ dài từ trung bình
- Đa dạng từ vựng (từ duy nhất / tổng từ)

## Phân tích cảm xúc

### Phân tích cảm xúc là gì?

- Xác định cảm xúc của văn bản: tích cực, tiêu cực hoặc trung tính.
- Ứng dụng: đánh giá khách hàng, giám sát mạng xã hội, danh tiếng thương hiệu.
- Các cách tiếp cận: dựa trên từ điển, machine learning, deep learning.

### Độ phân cực & Tính chủ quan

- **Polarity**: từ -1 (tiêu cực) đến +1 (tích cực)
- **Subjectivity**: từ 0 (khách quan) đến 1 (chủ quan)
- Cả hai số liệu đều cung cấp sự hiểu biết sâu sắc về văn bản.

### Thách thức trong phân tích cảm xúc

- Châm biếm và mỉa mai khó phát hiện.
- Phụ thuộc ngữ cảnh: "This movie was sick!" (tiếng lóng tích cực)
- Xử lý phủ định: "not good" vs "good"
- Ngôn ngữ đặc thù lĩnh vực và sự khác biệt văn hóa.

## Những hiểu biết quan trọng

- Phân tích tần suất đơn giản tiết lộ những hiểu biết đáng ngạc nhiên về tài liệu.
- N-grams nắm bắt ý nghĩa mà các từ riêng lẻ bỏ lỡ.
- Phân tích cảm xúc mạnh mẽ nhưng không hoàn hảo - luôn xác thực kết quả.
- Kết hợp nhiều kỹ thuật phân tích cung cấp sự hiểu biết phong phú hơn.

---

# **Thực hành Lab**

## Lab 1: Phân tích tần suất từ

```python
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

text = """
Natural Language Processing is a subfield of artificial intelligence.
It focuses on the interaction between computers and human language.
NLP enables machines to understand and generate human language.
"""

# Tiền xử lý
tokens = word_tokenize(text.lower())
stop_words = set(stopwords.words('english'))
tokens = [word for word in tokens 
          if word not in stop_words 
          and word not in string.punctuation]

# Phân tích tần suất
freq_dist = Counter(tokens)
print("Top 10 từ phổ biến nhất:")
for word, count in freq_dist.most_common(10):
    print(f"{word}: {count}")
```

## Lab 2: Phân tích N-gram

```python
from nltk import ngrams
from collections import Counter

def get_ngrams(text, n):
    tokens = word_tokenize(text.lower())
    n_grams = list(ngrams(tokens, n))
    return n_grams

text = "Natural language processing enables machine learning applications"

# Tạo bigrams
bigrams = get_ngrams(text, 2)
print("Bigrams:", bigrams)

# Tạo trigrams
trigrams = get_ngrams(text, 3)
print("Trigrams:", trigrams)

# Bigrams phổ biến nhất
bigram_freq = Counter(bigrams)
print("\nBigrams phổ biến nhất:", bigram_freq.most_common(5))
```

## Lab 3: Thống kê văn bản

```python
def calculate_text_statistics(text):
    # Tokenize
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    
    # Tính thống kê
    stats = {
        'tổng_từ': len(words),
        'từ_duy_nhất': len(set(words)),
        'tổng_câu': len(sentences),
        'độ_dài_từ_tb': sum(len(word) for word in words) / len(words),
        'độ_dài_câu_tb': len(words) / len(sentences),
        'đa_dạng_từ_vựng': len(set(words)) / len(words)
    }
    
    return stats

sample_text = """
Natural Language Processing is fascinating. It enables computers to understand human language.
NLP has many practical applications in today's technology-driven world.
"""

stats = calculate_text_statistics(sample_text)
for key, value in stats.items():
    print(f"{key}: {value:.2f}")
```

## Lab 4: Phân tích cảm xúc với TextBlob

```python
from textblob import TextBlob

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    
    # Xác định loại cảm xúc
    if sentiment.polarity > 0.1:
        category = "Tích cực"
    elif sentiment.polarity < -0.1:
        category = "Tiêu cực"
    else:
        category = "Trung tính"
    
    return {
        'polarity': sentiment.polarity,
        'subjectivity': sentiment.subjectivity,
        'category': category
    }

# Kiểm tra với các câu khác nhau
texts = [
    "I love this product! It's amazing and works perfectly.",
    "This is the worst experience I've ever had.",
    "The item arrived on time and matches the description.",
    "I'm not sure if I like this or not."
]

for text in texts:
    result = analyze_sentiment(text)
    print(f"\nVăn bản: {text}")
    print(f"Polarity: {result['polarity']:.2f}")
    print(f"Subjectivity: {result['subjectivity']:.2f}")
    print(f"Cảm xúc: {result['category']}")
```

## Lab 5: Phân tích cảm xúc với VADER

```python
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

texts = [
    "I absolutely love this! Best purchase ever!!!",
    "This product is terrible. Complete waste of money.",
    "It's okay, nothing special.",
    "Not bad, but could be better."
]

for text in texts:
    scores = sia.polarity_scores(text)
    print(f"\nVăn bản: {text}")
    print(f"Điểm: {scores}")
    
    # Xác định cảm xúc tổng thể
    if scores['compound'] >= 0.05:
        sentiment = "Tích cực"
    elif scores['compound'] <= -0.05:
        sentiment = "Tiêu cực"
    else:
        sentiment = "Trung tính"
    print(f"Tổng thể: {sentiment}")
```

---

# **Bài tập thực hành**

1. Phân tích tần suất từ trong cuốn sách hoặc bài báo yêu thích của bạn
2. Trích xuất và phân tích bigrams/trigrams từ bài đăng trên mạng xã hội
3. Xây dựng bộ phân loại cảm xúc cho đánh giá sản phẩm
4. So sánh kết quả cảm xúc từ TextBlob và VADER
5. Tạo trực quan hóa cho tần suất từ và phân phối cảm xúc
