---
title: "Day 41 - NLP Fundamentals Revision"
weight: 1
chapter: false
pre: "<b> 1.9.1. </b>"
---

**Date:** 2025-11-03   
**Status:** "Done"  

---

# **Lecture Notes**

## Week 8 Recap

### Core NLP Concepts Reviewed

**Text Preprocessing Pipeline**
- Tokenization: word and sentence level
- Normalization: lowercasing, removing special characters
- Stop word removal: context-dependent decisions
- Stemming vs Lemmatization: trade-offs and use cases

**Key Takeaways from Week 8**
- Preprocessing quality directly impacts model performance
- Different tasks require different preprocessing strategies
- Lemmatization generally preferred for production systems
- Always validate preprocessing choices against your specific use case

## Deep Dive: Tokenization Techniques

### Advanced Tokenization

**Subword Tokenization**
- Byte Pair Encoding (BPE): handles out-of-vocabulary words
- WordPiece: used by BERT and other transformers
- SentencePiece: language-agnostic tokenization
- Use case: multilingual models and rare word handling

**Comparison of Approaches**

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| Word | Simple, fast | Large vocabulary | Simple tasks |
| Character | No OOV issues | Long sequences | Spelling tasks |
| Subword | Balanced approach | More complex | Modern NLP |

### Regular Expression Tokenization

- Custom patterns for domain-specific text
- Handling URLs, emails, hashtags
- Preserving important punctuation
- Medical/technical text tokenization

## Text Normalization Best Practices

### When to Normalize

**Should Normalize:**
- Case sensitivity isn't important
- Consistent format needed
- Memory/computation constraints

**Avoid Over-normalization:**
- Named entities are important
- Sentiment analysis (emoticons matter)
- Code or technical documentation

### Unicode Handling

- Proper encoding/decoding
- Normalization forms (NFC, NFD, NFKC, NFKD)
- Handling multilingual text
- Emoji and special character preservation

## Key Insights

- Revisiting fundamentals reveals deeper understanding
- Edge cases often determine preprocessing strategy
- Modern NLP increasingly uses subword tokenization
- Balance between simplicity and effectiveness is crucial

---

# **Hands-On Labs**

## Lab 1: Comprehensive Preprocessing Comparison

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import string

text = """
The AI-powered system's performance improved significantly! 
Running multiple tests @ 99.9% accuracy. #MachineLearning
"""

def compare_preprocessing(text):
    """Compare different preprocessing approaches"""
    
    # Original
    print("ORIGINAL TEXT:")
    print(text)
    print("\n" + "="*60 + "\n")
    
    # Basic tokenization
    words_basic = word_tokenize(text)
    print("BASIC TOKENIZATION:")
    print(words_basic)
    print(f"Token count: {len(words_basic)}\n")
    
    # Lowercase + punctuation removal
    words_clean = [w.lower() for w in words_basic if w not in string.punctuation]
    print("LOWERCASE + NO PUNCTUATION:")
    print(words_clean)
    print(f"Token count: {len(words_clean)}\n")
    
    # Stop word removal
    stop_words = set(stopwords.words('english'))
    words_no_stop = [w for w in words_clean if w not in stop_words]
    print("NO STOP WORDS:")
    print(words_no_stop)
    print(f"Token count: {len(words_no_stop)}\n")
    
    # Stemming
    stemmer = PorterStemmer()
    words_stemmed = [stemmer.stem(w) for w in words_no_stop]
    print("STEMMED:")
    print(words_stemmed)
    print(f"Token count: {len(words_stemmed)}\n")
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words_lemmatized = [lemmatizer.lemmatize(w, pos='v') for w in words_no_stop]
    print("LEMMATIZED:")
    print(words_lemmatized)
    print(f"Token count: {len(words_lemmatized)}\n")

compare_preprocessing(text)
```

## Lab 2: Custom Tokenization with Regex

```python
import re

def custom_tokenize(text):
    """Custom tokenization preserving special patterns"""
    
    # Pattern for various text elements
    patterns = [
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',  # URLs
        r'[\w.+-]+@[\w-]+\.[\w.-]+',  # Emails
        r'#\w+',  # Hashtags
        r'@\w+',  # Mentions
        r'\d+\.?\d*%',  # Percentages
        r'\$\d+\.?\d*',  # Money
        r'\d{4}-\d{2}-\d{2}',  # Dates
        r'\w+',  # Words
        r'[^\w\s]',  # Punctuation
    ]
    
    pattern = '|'.join(patterns)
    tokens = re.findall(pattern, text)
    
    return tokens

# Test custom tokenization
sample = """
Check out https://example.com for AI updates!
Contact: info@ai-company.com
Event: 2025-11-03 @10:00AM
Budget: $50,000 (99.5% funded) #TechEvent
"""

tokens = custom_tokenize(sample)
print("Custom Tokenization:")
for i, token in enumerate(tokens, 1):
    print(f"{i}. {token}")
```

## Lab 3: Handling Multilingual Text

```python
def process_multilingual_text(text):
    """Process text with mixed languages and special characters"""
    
    # Unicode normalization
    import unicodedata
    
    # NFC (Canonical Decomposition, followed by Canonical Composition)
    normalized = unicodedata.normalize('NFC', text)
    
    # Identify language-specific patterns
    has_emoji = bool(re.search(r'[\U0001F600-\U0001F64F]', text))
    has_cjk = bool(re.search(r'[\u4e00-\u9fff]', text))  # Chinese/Japanese/Korean
    has_arabic = bool(re.search(r'[\u0600-\u06ff]', text))
    
    info = {
        'original': text,
        'normalized': normalized,
        'has_emoji': has_emoji,
        'has_cjk': has_cjk,
        'has_arabic': has_arabic,
        'length': len(text),
        'normalized_length': len(normalized)
    }
    
    return info

# Test multilingual processing
multilingual_samples = [
    "Hello 世界! 🌍",
    "Natural Language Processing",
    "café résumé naïve",
    "مرحبا العالم"
]

for sample in multilingual_samples:
    result = process_multilingual_text(sample)
    print(f"\nText: {result['original']}")
    print(f"Normalized: {result['normalized']}")
    print(f"Has Emoji: {result['has_emoji']}")
    print(f"Has CJK: {result['has_cjk']}")
    print(f"Has Arabic: {result['has_arabic']}")
```

## Lab 4: Preprocessing Pipeline Builder

```python
class PreprocessingPipeline:
    """Flexible preprocessing pipeline"""
    
    def __init__(self):
        self.steps = []
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def add_lowercase(self):
        self.steps.append(('lowercase', lambda x: x.lower()))
        return self
    
    def add_tokenization(self):
        self.steps.append(('tokenize', word_tokenize))
        return self
    
    def add_remove_punctuation(self):
        def remove_punct(tokens):
            return [t for t in tokens if t not in string.punctuation]
        self.steps.append(('remove_punct', remove_punct))
        return self
    
    def add_remove_stopwords(self):
        def remove_stop(tokens):
            return [t for t in tokens if t not in self.stop_words]
        self.steps.append(('remove_stop', remove_stop))
        return self
    
    def add_stemming(self):
        def stem(tokens):
            return [self.stemmer.stem(t) for t in tokens]
        self.steps.append(('stem', stem))
        return self
    
    def add_lemmatization(self, pos='v'):
        def lemmatize(tokens):
            return [self.lemmatizer.lemmatize(t, pos=pos) for t in tokens]
        self.steps.append(('lemmatize', lemmatize))
        return self
    
    def process(self, text):
        result = text
        for step_name, step_func in self.steps:
            result = step_func(result)
            print(f"After {step_name}: {result[:100]}...")  # Show first 100 chars
        return result

# Build custom pipeline
pipeline = (PreprocessingPipeline()
            .add_lowercase()
            .add_tokenization()
            .add_remove_punctuation()
            .add_remove_stopwords()
            .add_lemmatization())

text = "The students are studying NLP concepts and building amazing applications!"
result = pipeline.process(text)
print(f"\nFinal result: {result}")
```

---

# **Practice Exercises**

1. Compare preprocessing approaches on different text types (tweets, articles, code)
2. Build a preprocessing pipeline for a specific domain (medical, legal, social media)
3. Implement custom tokenization for handling special formats (product codes, IDs)
4. Analyze the impact of different preprocessing choices on a classification task
5. Create a preprocessing benchmark comparing speed and quality trade-offs
