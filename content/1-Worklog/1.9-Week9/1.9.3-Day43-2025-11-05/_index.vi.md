---
title: "Ngày 43 - Hệ thống phân tích cảm xúc Production"
weight: 3
chapter: false
pre: "<b> 1.9.3. </b>"
---

**Ngày:** 2025-11-05   
**Trạng thái:** "Hoàn thành"  

---

# **Ghi chú bài giảng**

## Ôn tập phân tích cảm xúc

### Các khái niệm chính từ Tuần 8

- Phân tích cảm xúc xác định cảm xúc: tích cực, tiêu cực, trung tính
- Các cách tiếp cận: dựa trên từ điển, machine learning, deep learning
- Số liệu: polarity (-1 đến +1), subjectivity (0 đến 1)
- Thách thức: châm biếm, ngữ cảnh, phủ định, ngôn ngữ đặc thù lĩnh vực

### Vượt xa phân tích cảm xúc cơ bản

**Phân tích cảm xúc dựa trên khía cạnh**
- Xác định cảm xúc hướng đến các khía cạnh/tính năng cụ thể
- Ví dụ: "Camera tuyệt vời nhưng pin tệ"
  - Camera: tích cực
  - Pin: tiêu cực
- Hiểu biết chi tiết hơn cho đánh giá sản phẩm

**Phát hiện cảm xúc**
- Vượt xa tích cực/tiêu cực: vui, giận, buồn, sợ, ngạc nhiên
- Vấn đề phân loại đa nhãn
- Hữu ích cho hỗ trợ khách hàng, giám sát sức khỏe tâm thần

## Hệ thống sẵn sàng Production

### Yêu cầu cho Production

**Khả năng mở rộng**
- Xử lý khối lượng yêu cầu cao
- Xử lý thời gian thực hoặc gần thời gian thực
- Xử lý hàng loạt cho tập dữ liệu lớn

**Độ mạnh mẽ**
- Xử lý văn bản nhiễu, không chính thức (lỗi chính tả, tiếng lóng, emoji)
- Xử lý graceful trên các trường hợp biên
- Điểm tin cậy rõ ràng

**Giám sát & Bảo trì**
- Theo dõi hiệu suất mô hình theo thời gian
- Phát hiện concept drift (ngôn ngữ thay đổi)
- A/B testing cho cải thiện mô hình
- Logging và theo dõi lỗi

### Thực hành tốt nhất về thiết kế API

**Xác thực đầu vào**
- Giới hạn độ dài văn bản
- Xử lý mã hóa ký tự
- Sanitization đầu vào độc hại

**Định dạng phản hồi**
```json
{
  "text": "văn bản đầu vào",
  "sentiment": "tích_cực",
  "confidence": 0.92,
  "polarity": 0.75,
  "subjectivity": 0.65,
  "timestamp": "2025-11-05T10:00:00Z"
}
```

**Xử lý lỗi**
- Thông báo lỗi có ý nghĩa
- Mã trạng thái HTTP
- Giới hạn tỷ lệ
- Xác thực yêu cầu

## Kỹ thuật nâng cao

### Transfer Learning

- Mô hình đã huấn luyện trước (BERT, RoBERTa) cho hiệu suất tốt hơn
- Fine-tuning trên dữ liệu đặc thù lĩnh vực
- Giảm yêu cầu dữ liệu huấn luyện
- Kết quả tiên tiến nhất

### Cách tiếp cận Ensemble

- Kết hợp phương pháp dựa trên từ điển và ML
- Voting hoặc trung bình có trọng số
- Cải thiện độ mạnh mẽ và độ chính xác
- Xử lý tốt hơn các trường hợp biên

## Những hiểu biết quan trọng

- Hệ thống production yêu cầu nhiều hơn độ chính xác tốt
- Thích ứng lĩnh vực quan trọng cho hiệu suất thực tế
- Giám sát liên tục thiết yếu để duy trì chất lượng
- Thiết kế API ảnh hưởng đến cả khả năng sử dụng và độ tin cậy hệ thống

---

# **Thực hành Lab**

## Lab 1: Phân tích cảm xúc dựa trên khía cạnh

```python
import spacy
from textblob import TextBlob
import re

nlp = spacy.load("en_core_web_sm")

class AspectSentimentAnalyzer:
    """Phân tích cảm xúc cho các khía cạnh cụ thể"""
    
    def __init__(self):
        self.aspect_keywords = {
            'camera': ['camera', 'photo', 'picture', 'lens', 'image'],
            'battery': ['battery', 'charge', 'power'],
            'screen': ['screen', 'display', 'brightness'],
            'performance': ['speed', 'fast', 'slow', 'lag', 'performance'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'value']
        }
    
    def extract_aspect_sentences(self, text):
        """Trích xuất câu đề cập đến mỗi khía cạnh"""
        doc = nlp(text)
        aspect_sentences = {aspect: [] for aspect in self.aspect_keywords}
        
        for sent in doc.sents:
            sent_text = sent.text.lower()
            for aspect, keywords in self.aspect_keywords.items():
                if any(keyword in sent_text for keyword in keywords):
                    aspect_sentences[aspect].append(sent.text)
        
        return aspect_sentences
    
    def analyze_aspect_sentiment(self, text):
        """Phân tích cảm xúc cho mỗi khía cạnh"""
        aspect_sentences = self.extract_aspect_sentences(text)
        results = {}
        
        for aspect, sentences in aspect_sentences.items():
            if not sentences:
                continue
            
            sentiments = []
            for sentence in sentences:
                blob = TextBlob(sentence)
                sentiments.append(blob.sentiment.polarity)
            
            avg_sentiment = sum(sentiments) / len(sentiments)
            results[aspect] = {
                'cảm_xúc': 'tích_cực' if avg_sentiment > 0.1 else 'tiêu_cực' if avg_sentiment < -0.1 else 'trung_tính',
                'polarity': avg_sentiment,
                'đề_cập': len(sentences),
                'ví_dụ': sentences[:2]  # 2 ví dụ đầu
            }
        
        return results

# Kiểm tra phân tích dựa trên khía cạnh
analyzer = AspectSentimentAnalyzer()

review = """
Điện thoại này có camera tuyệt vời! Ảnh rất rõ nét và sống động.
Tuy nhiên, pin thì thất vọng - chỉ dùng được chưa đầy một ngày.
Màn hình đẹp với độ sáng xuất sắc.
Hiệu suất tốt nhưng không tuyệt vời, đôi khi bị lag.
Với mức giá này, nó là giá trị tốt.
"""

aspects = analyzer.analyze_aspect_sentiment(review)

print("Phân tích cảm xúc dựa trên khía cạnh:\n")
for aspect, info in aspects.items():
    print(f"{aspect.upper()}:")
    print(f"  Cảm xúc: {info['cảm_xúc']}")
    print(f"  Polarity: {info['polarity']:.2f}")
    print(f"  Đề cập: {info['đề_cập']}")
    print(f"  Ví dụ: {info['ví_dụ']}")
    print()
```

## Lab 2: Phát hiện cảm xúc đa lớp

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

# Tập dữ liệu mẫu với nhiều cảm xúc
texts = [
    "Tôi rất vui và phấn khích về điều này!",
    "Điều này làm tôi rất tức giận và bực bội",
    "Tôi sợ hãi và lo lắng về tương lai",
    "Thật là một bất ngờ thú vị! Tôi rất vui",
    "Đây là tin buồn và thất vọng",
] * 10

# Cảm xúc đa nhãn (có thể có nhiều cảm xúc)
emotions = [
    {'vui': 1, 'giận': 0, 'sợ': 0, 'ngạc_nhiên': 0, 'buồn': 0},
    {'vui': 0, 'giận': 1, 'sợ': 0, 'ngạc_nhiên': 0, 'buồn': 0},
    {'vui': 0, 'giận': 0, 'sợ': 1, 'ngạc_nhiên': 0, 'buồn': 0},
    {'vui': 1, 'giận': 0, 'sợ': 0, 'ngạc_nhiên': 1, 'buồn': 0},
    {'vui': 0, 'giận': 0, 'sợ': 0, 'ngạc_nhiên': 0, 'buồn': 1},
] * 10

# Chuyển đổi sang định dạng array
emotion_labels = ['vui', 'giận', 'sợ', 'ngạc_nhiên', 'buồn']
y = np.array([[e[label] for label in emotion_labels] for e in emotions])

# Vector hóa
vectorizer = TfidfVectorizer(max_features=100)
X = vectorizer.fit_transform(texts)

# Bộ phân loại đa đầu ra
classifier = MultiOutputClassifier(LogisticRegression(max_iter=1000))
classifier.fit(X, y)

# Kiểm tra
test_texts = [
    "Tôi phấn khích và ngạc nhiên!",
    "Điều này đáng sợ và làm tôi lo lắng",
    "Rất thất vọng và buồn bã"
]

test_X = vectorizer.transform(test_texts)
predictions = classifier.predict(test_X)

print("Kết quả phát hiện cảm xúc:\n")
for text, pred in zip(test_texts, predictions):
    detected_emotions = [emotion_labels[i] for i, val in enumerate(pred) if val == 1]
    print(f"Văn bản: {text}")
    print(f"Cảm xúc: {', '.join(detected_emotions) if detected_emotions else 'trung_tính'}\n")
```

## Lab 3: API sẵn sàng Production với FastAPI

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from textblob import TextBlob
from datetime import datetime
import uvicorn

app = FastAPI(title="API Phân tích Cảm xúc")

class SentimentRequest(BaseModel):
    text: str
    
    @validator('text')
    def validate_text(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Văn bản không thể trống')
        if len(v) > 5000:
            raise ValueError('Văn bản quá dài (tối đa 5000 ký tự)')
        return v

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    polarity: float
    subjectivity: float
    timestamp: str

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """Phân tích cảm xúc của văn bản đầu vào"""
    try:
        blob = TextBlob(request.text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Xác định danh mục cảm xúc
        if polarity > 0.1:
            sentiment = "tích_cực"
            confidence = min(polarity, 1.0)
        elif polarity < -0.1:
            sentiment = "tiêu_cực"
            confidence = min(abs(polarity), 1.0)
        else:
            sentiment = "trung_tính"
            confidence = 1.0 - abs(polarity)
        
        return SentimentResponse(
            text=request.text,
            sentiment=sentiment,
            confidence=round(confidence, 2),
            polarity=round(polarity, 2),
            subjectivity=round(subjectivity, 2),
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Phân tích thất bại: {str(e)}")

@app.get("/health")
async def health_check():
    """Endpoint kiểm tra sức khỏe"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# Chạy: uvicorn script_name:app --reload
# Kiểm tra: curl -X POST "http://localhost:8000/analyze" -H "Content-Type: application/json" -d '{"text":"Tôi yêu sản phẩm này!"}'
```

## Lab 4: Bộ phân tích cảm xúc Ensemble

```python
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon', quiet=True)

class EnsembleSentimentAnalyzer:
    """Kết hợp nhiều phương pháp phân tích cảm xúc"""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
    
    def textblob_sentiment(self, text):
        """Phân tích TextBlob"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return 'tích_cực', polarity
        elif polarity < -0.1:
            return 'tiêu_cực', polarity
        else:
            return 'trung_tính', polarity
    
    def vader_sentiment(self, text):
        """Phân tích VADER"""
        scores = self.sia.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            return 'tích_cực', compound
        elif compound <= -0.05:
            return 'tiêu_cực', compound
        else:
            return 'trung_tính', compound
    
    def ensemble_predict(self, text, method='voting'):
        """Kết hợp dự đoán từ nhiều phương pháp"""
        
        # Lấy dự đoán từ cả hai phương pháp
        tb_sentiment, tb_score = self.textblob_sentiment(text)
        vader_sentiment, vader_score = self.vader_sentiment(text)
        
        if method == 'voting':
            # Voting đa số
            sentiments = [tb_sentiment, vader_sentiment]
            final_sentiment = max(set(sentiments), key=sentiments.count)
            
        elif method == 'weighted':
            # Trung bình có trọng số (trọng số cao hơn cho độ tin cậy cao hơn)
            tb_weight = abs(tb_score)
            vader_weight = abs(vader_score)
            total_weight = tb_weight + vader_weight
            
            if total_weight == 0:
                return 'trung_tính', 0.0, {'textblob': tb_sentiment, 'vader': vader_sentiment}
            
            # Trọng số cảm xúc
            sentiment_scores = {'tích_cực': 0, 'tiêu_cực': 0, 'trung_tính': 0}
            sentiment_scores[tb_sentiment] += tb_weight
            sentiment_scores[vader_sentiment] += vader_weight
            
            final_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            confidence = sentiment_scores[final_sentiment] / total_weight
            
            return final_sentiment, confidence, {
                'textblob': (tb_sentiment, tb_score),
                'vader': (vader_sentiment, vader_score)
            }
        
        return final_sentiment, 0.0, {'textblob': tb_sentiment, 'vader': vader_sentiment}

# Kiểm tra ensemble
ensemble = EnsembleSentimentAnalyzer()

test_sentences = [
    "Sản phẩm này tuyệt vời!",
    "Mua hàng tồi tệ nhất, lãng phí tiền",
    "Tạm được, không có gì đặc biệt",
    "Không tệ, nhưng có thể tốt hơn",
    "Tôi yêu nó quá!!!"
]

print("Phân tích cảm xúc Ensemble:\n")
for sentence in test_sentences:
    sentiment, confidence, details = ensemble.ensemble_predict(sentence, method='weighted')
    print(f"Văn bản: {sentence}")
    print(f"Cảm xúc cuối cùng: {sentiment} (tin cậy: {confidence:.2f})")
    print(f"TextBlob: {details['textblob']}")
    print(f"VADER: {details['vader']}")
    print()
```

## Lab 5: Phân tích cảm xúc với giám sát hiệu suất

```python
import time
from collections import defaultdict
from datetime import datetime

class MonitoredSentimentAnalyzer:
    """Bộ phân tích cảm xúc với giám sát tích hợp"""
    
    def __init__(self):
        self.stats = defaultdict(int)
        self.response_times = []
        self.errors = []
    
    def analyze(self, text):
        """Phân tích với giám sát"""
        start_time = time.time()
        
        try:
            # Thực hiện phân tích
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                sentiment = "tích_cực"
            elif polarity < -0.1:
                sentiment = "tiêu_cực"
            else:
                sentiment = "trung_tính"
            
            # Ghi lại thống kê
            self.stats['tổng_yêu_cầu'] += 1
            self.stats[f'{sentiment}_count'] += 1
            
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            
            return {
                'cảm_xúc': sentiment,
                'polarity': polarity,
                'thời_gian_phản_hồi': response_time
            }
        
        except Exception as e:
            self.stats['lỗi'] += 1
            self.errors.append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'text_length': len(text)
            })
            raise
    
    def get_metrics(self):
        """Lấy số liệu hiệu suất"""
        if not self.response_times:
            return {}
        
        return {
            'tổng_yêu_cầu': self.stats['tổng_yêu_cầu'],
            'tỷ_lệ_tích_cực': self.stats['tích_cực_count'] / self.stats['tổng_yêu_cầu'],
            'tỷ_lệ_tiêu_cực': self.stats['tiêu_cực_count'] / self.stats['tổng_yêu_cầu'],
            'tỷ_lệ_trung_tính': self.stats['trung_tính_count'] / self.stats['tổng_yêu_cầu'],
            'thời_gian_phản_hồi_tb': sum(self.response_times) / len(self.response_times),
            'thời_gian_phản_hồi_min': min(self.response_times),
            'thời_gian_phản_hồi_max': max(self.response_times),
            'số_lỗi': self.stats['lỗi'],
            'tỷ_lệ_lỗi': self.stats['lỗi'] / self.stats['tổng_yêu_cầu'] if self.stats['tổng_yêu_cầu'] > 0 else 0
        }

# Kiểm tra giám sát
monitored = MonitoredSentimentAnalyzer()

# Mô phỏng yêu cầu
test_texts = [
    "Sản phẩm tuyệt vời!",
    "Dịch vụ tệ",
    "Tạm được",
    "Yêu nó!",
    "Thất vọng"
] * 20

for text in test_texts:
    result = monitored.analyze(text)

# Lấy số liệu
metrics = monitored.get_metrics()
print("Số liệu hiệu suất:\n")
for key, value in metrics.items():
    if 'thời_gian' in key:
        print(f"{key}: {value*1000:.2f}ms")
    elif 'tỷ_lệ' in key:
        print(f"{key}: {value*100:.2f}%")
    else:
        print(f"{key}: {value}")
```

---

# **Bài tập thực hành**

1. Xây dựng bộ phân tích cảm xúc dựa trên khía cạnh cho đánh giá nhà hàng
2. Tạo dashboard cảm xúc thời gian thực bằng Streamlit
3. Triển khai phân tích xu hướng cảm xúc theo thời gian
4. Xây dựng API cảm xúc với giới hạn tỷ lệ và caching
5. So sánh các thư viện và cách tiếp cận phân tích cảm xúc khác nhau
