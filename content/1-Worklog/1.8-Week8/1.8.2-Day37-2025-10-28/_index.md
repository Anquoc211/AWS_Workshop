---
title: "Day 37 - Advanced Text Preprocessing"
weight: 2
chapter: false
pre: "<b> 1.8.2. </b>"
---

**Date:** 2025-10-28 
**Status:** "Done"  

---

# **Lecture Notes**

## Stop Word Removal

### Understanding Stop Words

- Common words with little semantic value: "the", "is", "at", "and", etc.
- Removing them reduces noise and computational cost.
- Context matters: for some tasks, stop words are important.

### When to Remove Stop Words

- Text classification: usually beneficial
- Sentiment analysis: be careful ("not" is crucial)
- Information retrieval: helps focus on content words
- Named entity recognition: may need context from stop words

## Stemming

### What is Stemming?

- Reducing words to their root form by removing suffixes.
- "running", "runs", "ran" → "run"
- Fast but can produce non-words (e.g., "studies" → "studi")

### Common Stemming Algorithms

- Porter Stemmer: most widely used, moderate accuracy
- Snowball Stemmer: improved version of Porter
- Lancaster Stemmer: most aggressive, may over-stem

## Lemmatization

### What is Lemmatization?

- Reducing words to their dictionary form (lemma).
- "running" → "run", "better" → "good"
- More accurate than stemming but slower.
- Requires part-of-speech information for best results.

### Stemming vs Lemmatization

| Aspect | Stemming | Lemmatization |
|--------|----------|---------------|
| Speed | Fast | Slower |
| Accuracy | Lower | Higher |
| Output | May not be real words | Always real words |
| Use case | Quick analysis | Accurate analysis |

## Key Insights

- Preprocessing choices depend on your specific task.
- Lemmatization generally better for production systems.
- Always validate preprocessing impact on model performance.
- Document your preprocessing pipeline for reproducibility.

---

# **Hands-On Labs**

## Lab 1: Stop Word Removal

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

text = "This is an example sentence demonstrating stop word removal."
words = word_tokenize(text.lower())

filtered_words = [word for word in words if word not in stop_words]

print("Original:", words)
print("Filtered:", filtered_words)
```

## Lab 2: Stemming Comparison

```python
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer

words = ["running", "runs", "ran", "easily", "fairly", "studies"]

porter = PorterStemmer()
snowball = SnowballStemmer('english')
lancaster = LancasterStemmer()

print("Word\t\tPorter\t\tSnowball\tLancaster")
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

print("Word\t\tLemma (verb)\tLemma (noun)")
print("-" * 50)
for word in words:
    verb_lemma = lemmatizer.lemmatize(word, pos=wordnet.VERB)
    noun_lemma = lemmatizer.lemmatize(word, pos=wordnet.NOUN)
    print(f"{word}\t\t{verb_lemma}\t\t{noun_lemma}")
```

## Lab 4: Complete Preprocessing Pipeline

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove punctuation and stop words
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
print("Processed tokens:", processed)
```

---

# **Practice Exercises**

1. Compare preprocessing results with and without stop word removal
2. Build a custom stop words list for a specific domain
3. Analyze which stemmer works best for your use case
4. Create a preprocessing pipeline function for tweets
