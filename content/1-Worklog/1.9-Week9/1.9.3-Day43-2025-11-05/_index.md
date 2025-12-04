---
title: "Day 43 - Production Sentiment Analysis Systems"
weight: 3
chapter: false
pre: "<b> 1.9.3. </b>"
---

**Date:** 2025-11-05   
**Status:** "Done"  

---

# **Lecture Notes**

## Sentiment Analysis Revision

### Key Concepts from Week 8

- Sentiment analysis determines emotional tone: positive, negative, neutral
- Approaches: lexicon-based, machine learning, deep learning
- Metrics: polarity (-1 to +1), subjectivity (0 to 1)
- Challenges: sarcasm, context, negation, domain-specific language

### Beyond Basic Sentiment

**Aspect-Based Sentiment Analysis**
- Identify sentiments toward specific aspects/features
- Example: "Great camera but terrible battery life"
  - Camera: positive
  - Battery: negative
- More granular insights for product reviews

**Emotion Detection**
- Beyond positive/negative: joy, anger, sadness, fear, surprise
- Multi-label classification problem
- Useful for customer support, mental health monitoring

## Production-Ready Systems

### Requirements for Production

**Scalability**
- Handle high request volumes
- Real-time or near real-time processing
- Batch processing for large datasets

**Robustness**
- Handle noisy, informal text (typos, slang, emojis)
- Graceful degradation on edge cases
- Clear confidence scores

**Monitoring & Maintenance**
- Track model performance over time
- Detect concept drift (language changes)
- A/B testing for model improvements
- Logging and error tracking

### API Design Best Practices

**Input Validation**
- Text length limits
- Character encoding handling
- Sanitization of malicious input

**Response Format**
```json
{
  "text": "input text",
  "sentiment": "positive",
  "confidence": 0.92,
  "polarity": 0.75,
  "subjectivity": 0.65,
  "timestamp": "2025-11-05T10:00:00Z"
}
```

**Error Handling**
- Meaningful error messages
- HTTP status codes
- Rate limiting
- Request validation

## Advanced Techniques

### Transfer Learning

- Pre-trained models (BERT, RoBERTa) for better performance
- Fine-tuning on domain-specific data
- Reduces training data requirements
- State-of-the-art results

### Ensemble Approaches

- Combine lexicon-based and ML methods
- Voting or weighted averaging
- Improves robustness and accuracy
- Better handling of edge cases

## Key Insights

- Production systems require more than just good accuracy
- Domain adaptation crucial for real-world performance
- Continuous monitoring essential for maintaining quality
- API design affects both usability and system reliability

---

# **Hands-On Labs**

## Lab 1: Aspect-Based Sentiment Analysis

```python
import spacy
from textblob import TextBlob
import re

nlp = spacy.load("en_core_web_sm")

class AspectSentimentAnalyzer:
    """Analyze sentiment for specific aspects"""
    
    def __init__(self):
        self.aspect_keywords = {
            'camera': ['camera', 'photo', 'picture', 'lens', 'image'],
            'battery': ['battery', 'charge', 'power'],
            'screen': ['screen', 'display', 'brightness'],
            'performance': ['speed', 'fast', 'slow', 'lag', 'performance'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'value']
        }
    
    def extract_aspect_sentences(self, text):
        """Extract sentences mentioning each aspect"""
        doc = nlp(text)
        aspect_sentences = {aspect: [] for aspect in self.aspect_keywords}
        
        for sent in doc.sents:
            sent_text = sent.text.lower()
            for aspect, keywords in self.aspect_keywords.items():
                if any(keyword in sent_text for keyword in keywords):
                    aspect_sentences[aspect].append(sent.text)
        
        return aspect_sentences
    
    def analyze_aspect_sentiment(self, text):
        """Analyze sentiment for each aspect"""
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
                'sentiment': 'positive' if avg_sentiment > 0.1 else 'negative' if avg_sentiment < -0.1 else 'neutral',
                'polarity': avg_sentiment,
                'mentions': len(sentences),
                'examples': sentences[:2]  # First 2 examples
            }
        
        return results

# Test aspect-based analysis
analyzer = AspectSentimentAnalyzer()

review = """
This phone has an amazing camera! The photos are crystal clear and vibrant.
However, the battery life is disappointing - it barely lasts a day.
The screen is gorgeous with excellent brightness.
Performance is good but not great, sometimes it lags.
For the price, it's a decent value.
"""

aspects = analyzer.analyze_aspect_sentiment(review)

print("Aspect-Based Sentiment Analysis:\n")
for aspect, info in aspects.items():
    print(f"{aspect.upper()}:")
    print(f"  Sentiment: {info['sentiment']}")
    print(f"  Polarity: {info['polarity']:.2f}")
    print(f"  Mentions: {info['mentions']}")
    print(f"  Examples: {info['examples']}")
    print()
```

## Lab 2: Multi-Class Emotion Detection

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

# Sample dataset with multiple emotions
texts = [
    "I'm so happy and excited about this!",
    "This makes me very angry and frustrated",
    "I'm scared and worried about the future",
    "What a pleasant surprise! I'm delighted",
    "This is sad and disappointing news",
] * 10

# Multi-label emotions (can have multiple emotions)
emotions = [
    {'joy': 1, 'anger': 0, 'fear': 0, 'surprise': 0, 'sadness': 0},
    {'joy': 0, 'anger': 1, 'fear': 0, 'surprise': 0, 'sadness': 0},
    {'joy': 0, 'anger': 0, 'fear': 1, 'surprise': 0, 'sadness': 0},
    {'joy': 1, 'anger': 0, 'fear': 0, 'surprise': 1, 'sadness': 0},
    {'joy': 0, 'anger': 0, 'fear': 0, 'surprise': 0, 'sadness': 1},
] * 10

# Convert to array format
emotion_labels = ['joy', 'anger', 'fear', 'surprise', 'sadness']
y = np.array([[e[label] for label in emotion_labels] for e in emotions])

# Vectorize
vectorizer = TfidfVectorizer(max_features=100)
X = vectorizer.fit_transform(texts)

# Multi-output classifier
classifier = MultiOutputClassifier(LogisticRegression(max_iter=1000))
classifier.fit(X, y)

# Test
test_texts = [
    "I'm thrilled and amazed!",
    "This is terrifying and makes me anxious",
    "So disappointed and upset"
]

test_X = vectorizer.transform(test_texts)
predictions = classifier.predict(test_X)

print("Emotion Detection Results:\n")
for text, pred in zip(test_texts, predictions):
    detected_emotions = [emotion_labels[i] for i, val in enumerate(pred) if val == 1]
    print(f"Text: {text}")
    print(f"Emotions: {', '.join(detected_emotions) if detected_emotions else 'neutral'}\n")
```

## Lab 3: Production-Ready API with FastAPI

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from textblob import TextBlob
from datetime import datetime
import uvicorn

app = FastAPI(title="Sentiment Analysis API")

class SentimentRequest(BaseModel):
    text: str
    
    @validator('text')
    def validate_text(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Text cannot be empty')
        if len(v) > 5000:
            raise ValueError('Text too long (max 5000 characters)')
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
    """Analyze sentiment of input text"""
    try:
        blob = TextBlob(request.text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine sentiment category
        if polarity > 0.1:
            sentiment = "positive"
            confidence = min(polarity, 1.0)
        elif polarity < -0.1:
            sentiment = "negative"
            confidence = min(abs(polarity), 1.0)
        else:
            sentiment = "neutral"
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
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# To run: uvicorn script_name:app --reload
# Test with: curl -X POST "http://localhost:8000/analyze" -H "Content-Type: application/json" -d '{"text":"I love this product!"}'
```

## Lab 4: Ensemble Sentiment Analyzer

```python
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon', quiet=True)

class EnsembleSentimentAnalyzer:
    """Combine multiple sentiment analysis methods"""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
    
    def textblob_sentiment(self, text):
        """TextBlob analysis"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return 'positive', polarity
        elif polarity < -0.1:
            return 'negative', polarity
        else:
            return 'neutral', polarity
    
    def vader_sentiment(self, text):
        """VADER analysis"""
        scores = self.sia.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            return 'positive', compound
        elif compound <= -0.05:
            return 'negative', compound
        else:
            return 'neutral', compound
    
    def ensemble_predict(self, text, method='voting'):
        """Combine predictions from multiple methods"""
        
        # Get predictions from both methods
        tb_sentiment, tb_score = self.textblob_sentiment(text)
        vader_sentiment, vader_score = self.vader_sentiment(text)
        
        if method == 'voting':
            # Majority voting
            sentiments = [tb_sentiment, vader_sentiment]
            final_sentiment = max(set(sentiments), key=sentiments.count)
            
        elif method == 'weighted':
            # Weighted average (give more weight to higher confidence)
            tb_weight = abs(tb_score)
            vader_weight = abs(vader_score)
            total_weight = tb_weight + vader_weight
            
            if total_weight == 0:
                return 'neutral', 0.0, {'textblob': tb_sentiment, 'vader': vader_sentiment}
            
            # Weight the sentiments
            sentiment_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
            sentiment_scores[tb_sentiment] += tb_weight
            sentiment_scores[vader_sentiment] += vader_weight
            
            final_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            confidence = sentiment_scores[final_sentiment] / total_weight
            
            return final_sentiment, confidence, {
                'textblob': (tb_sentiment, tb_score),
                'vader': (vader_sentiment, vader_score)
            }
        
        return final_sentiment, 0.0, {'textblob': tb_sentiment, 'vader': vader_sentiment}

# Test ensemble
ensemble = EnsembleSentimentAnalyzer()

test_sentences = [
    "This product is absolutely amazing!",
    "Worst purchase ever, total waste of money",
    "It's okay, nothing special",
    "Not bad, but could be better",
    "I love it so much!!!"
]

print("Ensemble Sentiment Analysis:\n")
for sentence in test_sentences:
    sentiment, confidence, details = ensemble.ensemble_predict(sentence, method='weighted')
    print(f"Text: {sentence}")
    print(f"Final Sentiment: {sentiment} (confidence: {confidence:.2f})")
    print(f"TextBlob: {details['textblob']}")
    print(f"VADER: {details['vader']}")
    print()
```

## Lab 5: Sentiment Analysis with Performance Monitoring

```python
import time
from collections import defaultdict
from datetime import datetime

class MonitoredSentimentAnalyzer:
    """Sentiment analyzer with built-in monitoring"""
    
    def __init__(self):
        self.stats = defaultdict(int)
        self.response_times = []
        self.errors = []
    
    def analyze(self, text):
        """Analyze with monitoring"""
        start_time = time.time()
        
        try:
            # Perform analysis
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                sentiment = "positive"
            elif polarity < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            # Record stats
            self.stats['total_requests'] += 1
            self.stats[f'{sentiment}_count'] += 1
            
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            
            return {
                'sentiment': sentiment,
                'polarity': polarity,
                'response_time': response_time
            }
        
        except Exception as e:
            self.stats['errors'] += 1
            self.errors.append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'text_length': len(text)
            })
            raise
    
    def get_metrics(self):
        """Get performance metrics"""
        if not self.response_times:
            return {}
        
        return {
            'total_requests': self.stats['total_requests'],
            'positive_ratio': self.stats['positive_count'] / self.stats['total_requests'],
            'negative_ratio': self.stats['negative_count'] / self.stats['total_requests'],
            'neutral_ratio': self.stats['neutral_count'] / self.stats['total_requests'],
            'avg_response_time': sum(self.response_times) / len(self.response_times),
            'min_response_time': min(self.response_times),
            'max_response_time': max(self.response_times),
            'error_count': self.stats['errors'],
            'error_rate': self.stats['errors'] / self.stats['total_requests'] if self.stats['total_requests'] > 0 else 0
        }

# Test monitoring
monitored = MonitoredSentimentAnalyzer()

# Simulate requests
test_texts = [
    "Great product!",
    "Terrible service",
    "It's okay",
    "Love it!",
    "Disappointed"
] * 20

for text in test_texts:
    result = monitored.analyze(text)

# Get metrics
metrics = monitored.get_metrics()
print("Performance Metrics:\n")
for key, value in metrics.items():
    if 'time' in key:
        print(f"{key}: {value*1000:.2f}ms")
    elif 'ratio' in key or 'rate' in key:
        print(f"{key}: {value*100:.2f}%")
    else:
        print(f"{key}: {value}")
```

---

# **Practice Exercises**

1. Build an aspect-based sentiment analyzer for restaurant reviews
2. Create a real-time sentiment dashboard using Streamlit
3. Implement sentiment trend analysis over time
4. Build a sentiment API with rate limiting and caching
5. Compare different sentiment analysis libraries and approaches
