---
title: "Day 36 - Introduction to NLP & Text Preprocessing"
weight: 1
chapter: false
pre: "<b> 1.8.1. </b>"
---

**Date:** 2025-10-27 
**Status:** "Done"  

---

# **Lecture Notes**

## Introduction to Natural Language Processing

### What is NLP?

- NLP enables computers to understand, interpret, and generate human language.
- Applications: chatbots, sentiment analysis, translation, voice assistants, search engines.
- Key challenges: ambiguity, context, idioms, multiple languages.

### NLP Pipeline Overview

```
Raw Text → Preprocessing → Feature Extraction → Model → Output
```

- Each stage transforms text into structured data for machine learning.
- Preprocessing is critical - "garbage in, garbage out" principle applies.

## Text Preprocessing Fundamentals

### Tokenization

- Breaking text into individual words or sentences.
- Word tokenization: "Hello world!" → ["Hello", "world", "!"]
- Sentence tokenization: Splitting paragraphs into sentences.
- Importance: foundation for all NLP tasks.

### Lowercasing

- Converting all text to lowercase for consistency.
- "Hello" and "hello" treated as the same word.
- Trade-off: may lose information (e.g., proper nouns).

### Removing Punctuation & Special Characters

- Cleaning text by removing non-alphanumeric characters.
- Keeps only meaningful words for analysis.
- Context-dependent: some punctuation may carry meaning.

## Key Insights

- NLP bridges the gap between human communication and machine understanding.
- Preprocessing quality directly impacts model performance.
- Different tasks require different preprocessing strategies.
- Always inspect your data before and after preprocessing.

---

# **Hands-On Labs**

## Lab 1: Basic Tokenization

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required data
nltk.download('punkt')

text = "Natural Language Processing is fascinating! It enables many applications."

# Word tokenization
words = word_tokenize(text)
print("Words:", words)

# Sentence tokenization
sentences = sent_tokenize(text)
print("Sentences:", sentences)
```

## Lab 2: Text Cleaning

```python
import re
import string

def clean_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

sample = "Hello World! This is NLP 101."
cleaned = clean_text(sample)
print("Original:", sample)
print("Cleaned:", cleaned)
```

## Lab 3: Exploring NLTK

```python
import nltk

# Explore NLTK capabilities
print("NLTK Version:", nltk.__version__)

# Download additional resources
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords

# View English stop words
stop_words = set(stopwords.words('english'))
print(f"Number of stop words: {len(stop_words)}")
print("Sample stop words:", list(stop_words)[:10])
```

---

# **Practice Exercises**

1. Tokenize a paragraph from your favorite book
2. Create a function that removes URLs from text
3. Compare word counts before and after preprocessing
4. Experiment with different tokenization methods
