---
title: "Ngày 40 - Xây dựng dự án NLP"
weight: 5
chapter: false
pre: "<b> 1.8.5. </b>"
---

**Ngày:** 2025-10-31  
**Trạng thái:** "Hoàn thành"  

---

# **Ghi chú bài giảng**

## Xây dựng hệ thống NLP Production

### Quy trình dự án NLP

```
Thu thập dữ liệu → Tiền xử lý → Kỹ thuật đặc trưng → 
Huấn luyện mô hình → Đánh giá → Triển khai → Giám sát
```

- Mỗi giai đoạn yêu cầu xem xét cẩn thận và lặp lại
- Chất lượng dữ liệu quyết định chất lượng mô hình
- Giám sát liên tục thiết yếu cho hệ thống production

### Các loại dự án NLP phổ biến

1. **Phân loại văn bản**: Phân loại tài liệu vào các lớp được xác định trước
2. **Chatbots**: AI đàm thoại cho dịch vụ khách hàng hoặc truy xuất thông tin
3. **Trích xuất thông tin**: Trích xuất dữ liệu có cấu trúc từ văn bản không có cấu trúc
4. **Tóm tắt văn bản**: Tạo bản tóm tắt ngắn gọn của tài liệu dài hơn
5. **Dịch máy**: Dịch văn bản giữa các ngôn ngữ

### Thực hành tốt nhất

- Bắt đầu đơn giản: mô hình cơ sở trước các giải pháp phức tạp
- Kiểm soát phiên bản dữ liệu và mô hình của bạn
- Ghi chép các quyết định tiền xử lý
- Kiểm tra trên dữ liệu thực tế, không chỉ tập dữ liệu sạch
- Giám sát hiệu suất mô hình trong production
- Có chiến lược dự phòng cho lỗi mô hình

## Kiến thức cơ bản về phân loại văn bản

### Cách tiếp cận

1. **Chuẩn bị dữ liệu**: Thu thập nhãn và làm sạch
2. **Trích xuất đặc trưng**: TF-IDF, word embeddings hoặc contextual embeddings
3. **Lựa chọn mô hình**: Naive Bayes, SVM hoặc neural networks
4. **Đánh giá**: Độ chính xác, precision, recall, F1-score
5. **Lặp lại**: Cải thiện dựa trên phân tích lỗi

### Các thuật toán phổ biến

- **Naive Bayes**: Nhanh, hoạt động tốt với dữ liệu hạn chế
- **SVM**: Hiệu quả cho văn bản với ranh giới rõ ràng
- **Logistic Regression**: Đơn giản và có thể giải thích
- **Random Forest**: Xử lý các mẫu phi tuyến
- **Neural Networks**: Hiệu suất tốt nhất nhưng cần nhiều dữ liệu hơn

## Kiến trúc Chatbot đơn giản

### Các thành phần

1. **Nhận dạng ý định**: Hiểu mục tiêu của người dùng
2. **Trích xuất thực thể**: Xác định thông tin chính
3. **Quản lý hội thoại**: Duy trì ngữ cảnh cuộc trò chuyện
4. **Tạo phản hồi**: Tạo câu trả lời phù hợp

### Các cách triển khai

- **Dựa trên quy tắc**: Khớp mẫu cho truy vấn đơn giản
- **Dựa trên truy xuất**: Chọn từ các phản hồi được xác định trước
- **Tạo sinh**: Tạo phản hồi bằng mô hình ngôn ngữ
- **Lai**: Kết hợp nhiều cách tiếp cận

## Những hiểu biết quan trọng

- Bắt đầu với các cách tiếp cận đơn giản và lặp lại
- Dữ liệu thực tế rất lộn xộn - tiền xử lý mạnh mẽ là quan trọng
- Khả năng giải thích mô hình quan trọng cho hệ thống production
- Trải nghiệm người dùng quan trọng không kém độ chính xác mô hình
- Luôn có tùy chọn human-in-the-loop cho ứng dụng quan trọng

---

# **Thực hành Lab**

## Lab 1: Phân loại văn bản - Phát hiện Spam

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Dữ liệu mẫu (trong thực tế, sử dụng tập dữ liệu thực)
emails = [
    ("Win a free iPhone now!", "spam"),
    ("Meeting tomorrow at 3pm", "ham"),
    ("Claim your prize money today", "spam"),
    ("Project update attached", "ham"),
    ("Get rich quick! Click here!", "spam"),
    ("Lunch plans for Friday?", "ham"),
    # Thêm nhiều ví dụ...
]

# Chuẩn bị dữ liệu
texts = [email[0] for email in emails]
labels = [email[1] for email in emails]

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Trích xuất đặc trưng
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Huấn luyện mô hình
classifier = MultinomialNB()
classifier.fit(X_train_vec, y_train)

# Dự đoán
y_pred = classifier.predict(X_test_vec)

# Đánh giá
print("Báo cáo phân loại:")
print(classification_report(y_test, y_pred))

# Kiểm tra email mới
new_email = ["Congratulations! You won $1000000"]
new_email_vec = vectorizer.transform(new_email)
prediction = classifier.predict(new_email_vec)
print(f"\nDự đoán email mới: {prediction[0]}")
```

## Lab 2: Bộ phân loại cảm xúc

```python
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Hàm tiền xử lý
def preprocess(text):
    # Chữ thường và tokenize
    tokens = word_tokenize(text.lower())
    
    # Loại bỏ stopwords và dấu câu
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens 
              if w not in stop_words 
              and w not in string.punctuation]
    
    return ' '.join(tokens)

# Đánh giá phim mẫu
reviews = [
    ("This movie was amazing! Loved every minute.", "positive"),
    ("Terrible film, waste of time and money.", "negative"),
    ("Great acting and storyline, highly recommend!", "positive"),
    ("Boring and predictable, wouldn't watch again.", "negative"),
    ("Masterpiece! One of the best films I've seen.", "positive"),
    # Thêm nhiều ví dụ...
]

# Tiền xử lý dữ liệu
texts = [preprocess(review[0]) for review in reviews]
labels = [review[1] for review in reviews]

# Chia và vector hóa
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Huấn luyện
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Đánh giá
accuracy = model.score(X_test_vec, y_test)
print(f"Độ chính xác: {accuracy:.2f}")

# Kiểm tra
test_review = "This movie exceeded my expectations, truly wonderful!"
test_vec = vectorizer.transform([preprocess(test_review)])
prediction = model.predict(test_vec)
print(f"\nĐánh giá: {test_review}")
print(f"Cảm xúc: {prediction[0]}")
```

## Lab 3: Chatbot đơn giản dựa trên quy tắc

```python
import re
import random

class SimpleChatbot:
    def __init__(self):
        self.patterns = {
            r'hi|hello|hey': [
                "Xin chào! Tôi có thể giúp gì cho bạn?",
                "Chào bạn! Tôi có thể làm gì cho bạn?",
                "Chào! Bạn khỏe không?"
            ],
            r'how are you': [
                "Tôi rất tốt, cảm ơn bạn đã hỏi!",
                "Tôi khỏe, còn bạn thì sao?",
                "Khỏe! Tôi có thể hỗ trợ bạn thế nào?"
            ],
            r'what is your name': [
                "Tôi là chatbot đơn giản được tạo để hỗ trợ bạn!",
                "Bạn có thể gọi tôi là ChatBot!",
                "Tôi là bot trợ lý thân thiện của bạn!"
            ],
            r'bye|goodbye|see you': [
                "Tạm biệt! Chúc bạn một ngày tốt lành!",
                "Hẹn gặp lại!",
                "Tạm biệt! Quay lại sớm nhé!"
            ],
            r'thank you|thanks': [
                "Không có gì!",
                "Rất vui được giúp đỡ!",
                "Bất cứ lúc nào!"
            ]
        }
        
        self.default_responses = [
            "Tôi không chắc tôi hiểu. Bạn có thể diễn đạt lại không?",
            "Bạn có thể làm rõ hơn không?",
            "Thú vị! Kể cho tôi nghe thêm."
        ]
    
    def get_response(self, user_input):
        user_input = user_input.lower()
        
        # Kiểm tra từng mẫu
        for pattern, responses in self.patterns.items():
            if re.search(pattern, user_input):
                return random.choice(responses)
        
        # Phản hồi mặc định
        return random.choice(self.default_responses)
    
    def chat(self):
        print("Chatbot: Xin chào! Tôi là chatbot đơn giản. Gõ 'quit' để thoát.")
        
        while True:
            user_input = input("Bạn: ")
            
            if user_input.lower() == 'quit':
                print("Chatbot: Tạm biệt!")
                break
            
            response = self.get_response(user_input)
            print(f"Chatbot: {response}")

# Kiểm tra chatbot
bot = SimpleChatbot()

# Chế độ tương tác
# bot.chat()

# Kiểm tra tin nhắn riêng lẻ
test_messages = [
    "Hello!",
    "How are you?",
    "What is your name?",
    "Thanks for your help!",
    "Tell me about NLP"
]

for message in test_messages:
    print(f"Bạn: {message}")
    print(f"Bot: {bot.get_response(message)}\n")
```

## Lab 4: Chatbot dựa trên ý định

```python
class IntentChatbot:
    def __init__(self):
        self.intents = {
            'chào_hỏi': {
                'patterns': ['hello', 'hi', 'hey', 'good morning', 'xin chào', 'chào'],
                'responses': ['Xin chào!', 'Chào bạn!', 'Chào mừng!']
            },
            'thời_tiết': {
                'patterns': ['weather', 'temperature', 'forecast', 'thời tiết', 'nhiệt độ'],
                'responses': [
                    'Tôi không thể kiểm tra thời tiết, nhưng bạn có thể thử ứng dụng thời tiết!',
                    'Để biết thông tin thời tiết, vui lòng kiểm tra dịch vụ thời tiết.'
                ]
            },
            'thời_gian': {
                'patterns': ['time', 'clock', 'what time', 'giờ', 'mấy giờ'],
                'responses': ['Vui lòng kiểm tra đồng hồ hệ thống của bạn.']
            },
            'khả_năng': {
                'patterns': ['what can you do', 'your features', 'help', 'bạn làm được gì', 'giúp'],
                'responses': [
                    'Tôi có thể trò chuyện với bạn, trả lời câu hỏi đơn giản và hỗ trợ các truy vấn cơ bản!',
                    'Tôi là chatbot đơn giản có thể có cuộc trò chuyện cơ bản.'
                ]
            }
        }
    
    def classify_intent(self, text):
        text = text.lower()
        
        for intent, data in self.intents.items():
            for pattern in data['patterns']:
                if pattern in text:
                    return intent
        
        return 'không_xác_định'
    
    def get_response(self, text):
        intent = self.classify_intent(text)
        
        if intent == 'không_xác_định':
            return "Tôi không chắc cách trả lời điều đó. Thử hỏi về thời tiết, thời gian hoặc tôi có thể làm gì!"
        
        return random.choice(self.intents[intent]['responses'])

# Kiểm tra
intent_bot = IntentChatbot()

queries = [
    "Xin chào!",
    "Thời tiết thế nào?",
    "Mấy giờ rồi?",
    "Bạn làm được gì?",
    "Kể cho tôi một câu chuyện cười"
]

for query in queries:
    print(f"Người dùng: {query}")
    print(f"Bot: {intent_bot.get_response(query)}")
    print(f"Ý định: {intent_bot.classify_intent(query)}\n")
```

## Lab 5: Dự án quy trình NLP hoàn chỉnh

```python
import spacy
from textblob import TextBlob
from collections import Counter

class TextAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
    
    def analyze(self, text):
        """Quy trình phân tích văn bản hoàn chỉnh"""
        
        # Thống kê cơ bản
        doc = self.nlp(text)
        word_count = len([token for token in doc if not token.is_punct])
        sentence_count = len(list(doc.sents))
        
        # Thực thể
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Cảm xúc
        blob = TextBlob(text)
        sentiment = {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
        
        # Từ khóa (các từ có ý nghĩa phổ biến nhất)
        keywords = Counter([token.lemma_.lower() for token in doc 
                          if not token.is_stop and not token.is_punct 
                          and token.pos_ in ['NOUN', 'VERB', 'ADJ']])
        
        return {
            'số_từ': word_count,
            'số_câu': sentence_count,
            'thực_thể': entities,
            'cảm_xúc': sentiment,
            'từ_khóa_hàng_đầu': keywords.most_common(5)
        }

# Kiểm tra
analyzer = TextAnalyzer()

sample_article = """
Apple Inc. thông báo hôm nay rằng Tim Cook sẽ phát biểu chính tại hội nghị hàng năm
ở Cupertino, California. Sự kiện sẽ giới thiệu các sản phẩm đổi mới bao gồm
iPhone mới và các bản cập nhật phần mềm mang tính cách mạng. Các chuyên gia ngành phấn khích
về tác động tiềm năng đến thị trường công nghệ trên toàn thế giới.
"""

results = analyzer.analyze(sample_article)

print("=== Kết quả phân tích văn bản ===\n")
print(f"Số từ: {results['số_từ']}")
print(f"Số câu: {results['số_câu']}")

print("\nThực thể được đặt tên:")
for entity, label in results['thực_thể']:
    print(f"  {entity} ({label})")

print(f"\nCảm xúc:")
print(f"  Polarity: {results['cảm_xúc']['polarity']:.2f}")
print(f"  Subjectivity: {results['cảm_xúc']['subjectivity']:.2f}")

print("\nTừ khóa hàng đầu:")
for word, count in results['từ_khóa_hàng_đầu']:
    print(f"  {word}: {count}")
```

---

# **Bài tập thực hành**

1. Xây dựng bộ phân loại bài báo cho các danh mục khác nhau
2. Tạo chatbot FAQ cho một lĩnh vực cụ thể
3. Triển khai công cụ tóm tắt văn bản
4. Xây dựng hệ thống trích xuất từ khóa
5. Tạo dashboard phân tích cảm xúc hoàn chỉnh

---

# **Tóm tắt tuần 8**

Tuần này bao gồm:
- Kiến thức cơ bản về NLP và tiền xử lý văn bản
- Tiền xử lý nâng cao (stemming, lemmatization)
- Phân tích văn bản và phân tích cảm xúc
- Nhận dạng thực thể được đặt tên với spaCy
- Xây dựng ứng dụng NLP thực tế

**Kỹ năng đã đạt được:**
- Quy trình tiền xử lý văn bản
- Triển khai phân tích cảm xúc
- Trích xuất và phân loại thực thể
- Xây dựng mô hình phân loại văn bản
- Phát triển chatbot đơn giản
