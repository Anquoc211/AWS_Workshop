---
title: "Day 38 - Text Analysis & Sentiment Analysis"
weight: 3
chapter: false
pre: "<b> 1.8.3. </b>"
---

**Date:** 2025-10-29
**Status:** "Done"  

---

# **Lecture Notes**

## Basic Text Analysis

### Word Frequency Analysis

- Counting how often each word appears in text.
- Identifies most common terms and themes.
- Foundation for many NLP applications.
- Useful for understanding document content quickly.

### N-grams

- Sequences of N consecutive words.
- Unigrams: single words ("machine")
- Bigrams: two words ("machine learning")
- Trigrams: three words ("natural language processing")
- Captures context and common phrases.

### Text Statistics

- Document length (words, sentences, characters)
- Vocabulary size (unique words)
- Average word length
- Lexical diversity (unique words / total words)

## Sentiment Analysis

### What is Sentiment Analysis?

- Determining emotional tone of text: positive, negative, or neutral.
- Applications: customer reviews, social media monitoring, brand reputation.
- Approaches: lexicon-based, machine learning, deep learning.

### Sentiment Polarity & Subjectivity

- **Polarity**: ranges from -1 (negative) to +1 (positive)
- **Subjectivity**: ranges from 0 (objective) to 1 (subjective)
- Both metrics provide nuanced understanding of text.

### Challenges in Sentiment Analysis

- Sarcasm and irony are difficult to detect.
- Context dependency: "This movie was sick!" (positive slang)
- Negation handling: "not good" vs "good"
- Domain-specific language and cultural differences.

## Key Insights

- Simple frequency analysis reveals surprising insights about documents.
- N-grams capture meaning that individual words miss.
- Sentiment analysis is powerful but imperfect - always validate results.
- Combining multiple analysis techniques provides richer understanding.

---

# **Hands-On Labs**

## Lab 1: Word Frequency Analysis

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

# Preprocess
tokens = word_tokenize(text.lower())
stop_words = set(stopwords.words('english'))
tokens = [word for word in tokens 
          if word not in stop_words 
          and word not in string.punctuation]

# Frequency analysis
freq_dist = Counter(tokens)
print("Top 10 most common words:")
for word, count in freq_dist.most_common(10):
    print(f"{word}: {count}")
```

## Lab 2: N-gram Analysis

```python
from nltk import ngrams
from collections import Counter

def get_ngrams(text, n):
    tokens = word_tokenize(text.lower())
    n_grams = list(ngrams(tokens, n))
    return n_grams

text = "Natural language processing enables machine learning applications"

# Generate bigrams
bigrams = get_ngrams(text, 2)
print("Bigrams:", bigrams)

# Generate trigrams
trigrams = get_ngrams(text, 3)
print("Trigrams:", trigrams)

# Most common bigrams
bigram_freq = Counter(bigrams)
print("\nMost common bigrams:", bigram_freq.most_common(5))
```

## Lab 3: Text Statistics

```python
def calculate_text_statistics(text):
    # Tokenize
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    
    # Calculate statistics
    stats = {
        'total_words': len(words),
        'unique_words': len(set(words)),
        'total_sentences': len(sentences),
        'avg_word_length': sum(len(word) for word in words) / len(words),
        'avg_sentence_length': len(words) / len(sentences),
        'lexical_diversity': len(set(words)) / len(words)
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

## Lab 4: Sentiment Analysis with TextBlob

```python
from textblob import TextBlob

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    
    # Determine sentiment category
    if sentiment.polarity > 0.1:
        category = "Positive"
    elif sentiment.polarity < -0.1:
        category = "Negative"
    else:
        category = "Neutral"
    
    return {
        'polarity': sentiment.polarity,
        'subjectivity': sentiment.subjectivity,
        'category': category
    }

# Test with different sentences
texts = [
    "I love this product! It's amazing and works perfectly.",
    "This is the worst experience I've ever had.",
    "The item arrived on time and matches the description.",
    "I'm not sure if I like this or not."
]

for text in texts:
    result = analyze_sentiment(text)
    print(f"\nText: {text}")
    print(f"Polarity: {result['polarity']:.2f}")
    print(f"Subjectivity: {result['subjectivity']:.2f}")
    print(f"Sentiment: {result['category']}")
```

## Lab 5: Sentiment Analysis with VADER

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
    print(f"\nText: {text}")
    print(f"Scores: {scores}")
    
    # Determine overall sentiment
    if scores['compound'] >= 0.05:
        sentiment = "Positive"
    elif scores['compound'] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    print(f"Overall: {sentiment}")
```

---

# **Practice Exercises**

1. Analyze word frequency in your favorite book or article
2. Extract and analyze bigrams/trigrams from social media posts
3. Build a sentiment classifier for product reviews
4. Compare TextBlob and VADER sentiment results
5. Create visualizations for word frequency and sentiment distribution
