---
title: "Day 40 - Building NLP Projects"
weight: 5
chapter: false
pre: "<b> 1.8.5. </b>"
---

**Date:** 2025-10-31
**Status:** "Done"  

---

# **Lecture Notes**

## Building Production NLP Systems

### NLP Project Workflow

```
Data Collection → Preprocessing → Feature Engineering → 
Model Training → Evaluation → Deployment → Monitoring
```

- Each stage requires careful consideration and iteration
- Data quality determines model quality
- Continuous monitoring essential for production systems

### Common NLP Project Types

1. **Text Classification**: Categorizing documents into predefined classes
2. **Chatbots**: Conversational AI for customer service or information retrieval
3. **Information Extraction**: Extracting structured data from unstructured text
4. **Text Summarization**: Creating concise summaries of longer documents
5. **Machine Translation**: Translating text between languages

### Best Practices

- Start simple: baseline models before complex solutions
- Version control your data and models
- Document preprocessing decisions
- Test on real-world data, not just clean datasets
- Monitor model performance in production
- Have fallback strategies for model failures

## Text Classification Fundamentals

### Approach

1. **Data preparation**: Label collection and cleaning
2. **Feature extraction**: TF-IDF, word embeddings, or contextual embeddings
3. **Model selection**: Naive Bayes, SVM, or neural networks
4. **Evaluation**: Accuracy, precision, recall, F1-score
5. **Iteration**: Improve based on error analysis

### Popular Algorithms

- **Naive Bayes**: Fast, works well with limited data
- **SVM**: Effective for text with clear margins
- **Logistic Regression**: Simple and interpretable
- **Random Forest**: Handles non-linear patterns
- **Neural Networks**: Best performance but requires more data

## Simple Chatbot Architecture

### Components

1. **Intent Recognition**: Understanding user's goal
2. **Entity Extraction**: Identifying key information
3. **Dialogue Management**: Maintaining conversation context
4. **Response Generation**: Creating appropriate replies

### Implementation Approaches

- **Rule-based**: Pattern matching for simple queries
- **Retrieval-based**: Select from predefined responses
- **Generative**: Generate responses using language models
- **Hybrid**: Combine multiple approaches

## Key Insights

- Start with simple approaches and iterate
- Real-world data is messy - robust preprocessing is critical
- Model interpretability matters for production systems
- User experience is as important as model accuracy
- Always have a human-in-the-loop option for critical applications

---

# **Hands-On Labs**

## Lab 1: Text Classification - Spam Detection

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Sample data (in practice, use real dataset)
emails = [
    ("Win a free iPhone now!", "spam"),
    ("Meeting tomorrow at 3pm", "ham"),
    ("Claim your prize money today", "spam"),
    ("Project update attached", "ham"),
    ("Get rich quick! Click here!", "spam"),
    ("Lunch plans for Friday?", "ham"),
    # Add more examples...
]

# Prepare data
texts = [email[0] for email in emails]
labels = [email[1] for email in emails]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
classifier = MultinomialNB()
classifier.fit(X_train_vec, y_train)

# Predict
y_pred = classifier.predict(X_test_vec)

# Evaluate
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Test new email
new_email = ["Congratulations! You won $1000000"]
new_email_vec = vectorizer.transform(new_email)
prediction = classifier.predict(new_email_vec)
print(f"\nNew email prediction: {prediction[0]}")
```

## Lab 2: Sentiment Classifier

```python
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Preprocessing function
def preprocess(text):
    # Lowercase and tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens 
              if w not in stop_words 
              and w not in string.punctuation]
    
    return ' '.join(tokens)

# Sample movie reviews
reviews = [
    ("This movie was amazing! Loved every minute.", "positive"),
    ("Terrible film, waste of time and money.", "negative"),
    ("Great acting and storyline, highly recommend!", "positive"),
    ("Boring and predictable, wouldn't watch again.", "negative"),
    ("Masterpiece! One of the best films I've seen.", "positive"),
    # Add more examples...
]

# Preprocess data
texts = [preprocess(review[0]) for review in reviews]
labels = [review[1] for review in reviews]

# Split and vectorize
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
accuracy = model.score(X_test_vec, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Test
test_review = "This movie exceeded my expectations, truly wonderful!"
test_vec = vectorizer.transform([preprocess(test_review)])
prediction = model.predict(test_vec)
print(f"\nReview: {test_review}")
print(f"Sentiment: {prediction[0]}")
```

## Lab 3: Simple Rule-Based Chatbot

```python
import re
import random

class SimpleChatbot:
    def __init__(self):
        self.patterns = {
            r'hi|hello|hey': [
                "Hello! How can I help you?",
                "Hi there! What can I do for you?",
                "Hey! How are you doing?"
            ],
            r'how are you': [
                "I'm doing great, thanks for asking!",
                "I'm well, how about you?",
                "Doing fine! How can I assist you?"
            ],
            r'what is your name': [
                "I'm a simple chatbot created to assist you!",
                "You can call me ChatBot!",
                "I'm your friendly assistant bot!"
            ],
            r'bye|goodbye|see you': [
                "Goodbye! Have a great day!",
                "See you later!",
                "Bye! Come back soon!"
            ],
            r'thank you|thanks': [
                "You're welcome!",
                "Happy to help!",
                "Anytime!"
            ]
        }
        
        self.default_responses = [
            "I'm not sure I understand. Can you rephrase that?",
            "Could you please elaborate?",
            "Interesting! Tell me more."
        ]
    
    def get_response(self, user_input):
        user_input = user_input.lower()
        
        # Check each pattern
        for pattern, responses in self.patterns.items():
            if re.search(pattern, user_input):
                return random.choice(responses)
        
        # Default response
        return random.choice(self.default_responses)
    
    def chat(self):
        print("Chatbot: Hello! I'm a simple chatbot. Type 'quit' to exit.")
        
        while True:
            user_input = input("You: ")
            
            if user_input.lower() == 'quit':
                print("Chatbot: Goodbye!")
                break
            
            response = self.get_response(user_input)
            print(f"Chatbot: {response}")

# Test the chatbot
bot = SimpleChatbot()

# Interactive mode
# bot.chat()

# Test individual messages
test_messages = [
    "Hello!",
    "How are you?",
    "What is your name?",
    "Thanks for your help!",
    "Tell me about NLP"
]

for message in test_messages:
    print(f"You: {message}")
    print(f"Bot: {bot.get_response(message)}\n")
```

## Lab 4: Intent-Based Chatbot

```python
class IntentChatbot:
    def __init__(self):
        self.intents = {
            'greeting': {
                'patterns': ['hello', 'hi', 'hey', 'good morning'],
                'responses': ['Hello!', 'Hi there!', 'Greetings!']
            },
            'weather': {
                'patterns': ['weather', 'temperature', 'forecast'],
                'responses': [
                    'I cannot check the weather, but you can try a weather app!',
                    'For weather info, please check a weather service.'
                ]
            },
            'time': {
                'patterns': ['time', 'clock', 'what time'],
                'responses': ['Please check your system clock for the current time.']
            },
            'capabilities': {
                'patterns': ['what can you do', 'your features', 'help'],
                'responses': [
                    'I can chat with you, answer simple questions, and assist with basic queries!',
                    'I\'m a simple chatbot that can have basic conversations.'
                ]
            }
        }
    
    def classify_intent(self, text):
        text = text.lower()
        
        for intent, data in self.intents.items():
            for pattern in data['patterns']:
                if pattern in text:
                    return intent
        
        return 'unknown'
    
    def get_response(self, text):
        intent = self.classify_intent(text)
        
        if intent == 'unknown':
            return "I'm not sure how to respond to that. Try asking about weather, time, or what I can do!"
        
        return random.choice(self.intents[intent]['responses'])

# Test
intent_bot = IntentChatbot()

queries = [
    "Hello!",
    "What's the weather like?",
    "What time is it?",
    "What can you do?",
    "Tell me a joke"
]

for query in queries:
    print(f"User: {query}")
    print(f"Bot: {intent_bot.get_response(query)}")
    print(f"Intent: {intent_bot.classify_intent(query)}\n")
```

## Lab 5: Complete NLP Pipeline Project

```python
import spacy
from textblob import TextBlob
from collections import Counter

class TextAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
    
    def analyze(self, text):
        """Complete text analysis pipeline"""
        
        # Basic stats
        doc = self.nlp(text)
        word_count = len([token for token in doc if not token.is_punct])
        sentence_count = len(list(doc.sents))
        
        # Entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Sentiment
        blob = TextBlob(text)
        sentiment = {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
        
        # Keywords (most common meaningful words)
        keywords = Counter([token.lemma_.lower() for token in doc 
                          if not token.is_stop and not token.is_punct 
                          and token.pos_ in ['NOUN', 'VERB', 'ADJ']])
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'entities': entities,
            'sentiment': sentiment,
            'top_keywords': keywords.most_common(5)
        }

# Test
analyzer = TextAnalyzer()

sample_article = """
Apple Inc. announced today that Tim Cook will keynote their annual conference
in Cupertino, California. The event will showcase innovative products including
the new iPhone and revolutionary software updates. Industry experts are excited
about the potential impact on technology markets worldwide.
"""

results = analyzer.analyze(sample_article)

print("=== Text Analysis Results ===\n")
print(f"Word Count: {results['word_count']}")
print(f"Sentence Count: {results['sentence_count']}")

print("\nNamed Entities:")
for entity, label in results['entities']:
    print(f"  {entity} ({label})")

print(f"\nSentiment:")
print(f"  Polarity: {results['sentiment']['polarity']:.2f}")
print(f"  Subjectivity: {results['sentiment']['subjectivity']:.2f}")

print("\nTop Keywords:")
for word, count in results['top_keywords']:
    print(f"  {word}: {count}")
```

---

# **Practice Exercises**

1. Build a news article classifier for different categories
2. Create a FAQ chatbot for a specific domain
3. Implement a text summarization tool
4. Build a keyword extraction system
5. Create a complete sentiment analysis dashboard

---

# **Week 8 Summary**

This week covered:
- NLP fundamentals and text preprocessing
- Advanced preprocessing (stemming, lemmatization)
- Text analysis and sentiment analysis
- Named Entity Recognition with spaCy
- Building practical NLP applications

**Key Skills Acquired:**
- Text preprocessing pipelines
- Sentiment analysis implementation
- Entity extraction and classification
- Text classification model building
- Simple chatbot development
