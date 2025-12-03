---
title: "Ngày 37 - Tiền xử lý văn bản nâng cao"
weight: 2
chapter: false
pre: "<b> 1.8.2. </b>"
---

**Ngày:** 2025-10-28 
**Trạng thái:** "Hoàn thành"  

---

# **Ghi chú bài giảng**

## Loại bỏ Stop Word

### Hiểu về Stop Words

- Các từ phổ biến có ít giá trị ngữ nghĩa: "the", "is", "at", "and", v.v.
- Loại bỏ chúng giúp giảm nhiễu và chi phí tính toán.
- Ngữ cảnh quan trọng: đối với một số tác vụ, stop words lại quan trọng.

### Khi nào nên loại bỏ Stop Words

- Phân loại văn bản: thường có lợi
- Phân tích cảm xúc: cẩn thận ("not" rất quan trọng)
- Truy xuất thông tin: giúp tập trung vào các từ nội dung
- Nhận dạng thực thể: có thể cần ngữ cảnh từ stop words

## Stemming

### Stemming là gì?

- Giảm từ về dạng gốc bằng cách loại bỏ hậu tố.
- "running", "runs", "ran" → "run"
- Nhanh nhưng có thể tạo ra các từ không có nghĩa (ví dụ: "studies" → "studi")

### Các thuật toán Stemming phổ biến

- Porter Stemmer: được sử dụng rộng rãi nhất, độ chính xác trung bình
- Snowball Stemmer: phiên bản cải tiến của Porter
- Lancaster Stemmer: mạnh mẽ nhất, có thể stem quá mức

## Lemmatization

### Lemmatization là gì?

- Giảm từ về dạng từ điển (lemma).
- "running" → "run", "better" → "good"
- Chính xác hơn stemming nhưng chậm hơn.
- Yêu cầu thông tin từ loại để có kết quả tốt nhất.

### Stemming vs Lemmatization

| Khía cạnh | Stemming | Lemmatization |
|-----------|----------|---------------|
| Tốc độ | Nhanh | Chậm hơn |
| Độ chính xác | Thấp hơn | Cao hơn |
| Đầu ra | Có thể không phải từ thật | Luôn là từ thật |
| Trường hợp sử dụng | Phân tích nhanh | Phân tích chính xác |

## Những hiểu biết quan trọng

- Lựa chọn tiền xử lý phụ thuộc vào tác vụ cụ thể của bạn.
- Lemmatization thường tốt hơn cho hệ thống production.
- Luôn xác thực tác động của tiền xử lý đến hiệu suất mô hình.
- Ghi chép quy trình tiền xử lý của bạn để có thể tái tạo.

---

# **Thực hành Lab**

## Lab 1: Loại bỏ Stop Word

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

text = "This is an example sentence demonstrating stop word removal."
words = word_tokenize(text.lower())

filtered_words = [word for word in words if word not in stop_words]

print("Gốc:", words)
print("Đã lọc:", filtered_words)
```

## Lab 2: So sánh Stemming

```python
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer

words = ["running", "runs", "ran", "easily", "fairly", "studies"]

porter = PorterStemmer()
snowball = SnowballStemmer('english')
lancaster = LancasterStemmer()

print("Từ\t\tPorter\t\tSnowball\tLancaster")
print("-" * 60)
for word in words:
    print(f"{word}\t\t{porter.stem(word)}\t\t{snowball.stem(word)}\t\t{lancaster.stem(word)}")
```

## Lab 3: Lemmatization

```python
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

words = ["running", "runs", "ran", "better", "studying", "studies"]

print("Từ\t\tLemma (động từ)\tLemma (danh từ)")
print("-" * 50)
for word in words:
    verb_lemma = lemmatizer.lemmatize(word, pos=wordnet.VERB)
    noun_lemma = lemmatizer.lemmatize(word, pos=wordnet.NOUN)
    print(f"{word}\t\t{verb_lemma}\t\t{noun_lemma}")
```

## Lab 4: Quy trình tiền xử lý hoàn chỉnh

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

def preprocess_text(text):
    # Chữ thường
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Loại bỏ dấu câu và stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens 
              if word not in string.punctuation 
              and word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens

sample = "The students are studying NLP concepts. They're learning quickly!"
processed = preprocess_text(sample)
print("Tokens đã xử lý:", processed)
```

---

# **Bài tập thực hành**

1. So sánh kết quả tiền xử lý có và không có loại bỏ stop word
2. Xây dựng danh sách stop words tùy chỉnh cho một lĩnh vực cụ thể
3. Phân tích stemmer nào hoạt động tốt nhất cho trường hợp sử dụng của bạn
4. Tạo hàm quy trình tiền xử lý cho tweets
