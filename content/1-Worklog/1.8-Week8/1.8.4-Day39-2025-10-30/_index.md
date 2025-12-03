---
title: "Day 39 - Named Entity Recognition (NER)"
weight: 4
chapter: false
pre: "<b> 1.8.4. </b>"
---

**Date:** 2025-10-30
**Status:** "Done"  

---

# **Lecture Notes**

## Named Entity Recognition (NER)

### What is NER?

- Identifying and classifying named entities in text.
- Entities: people, organizations, locations, dates, products, etc.
- Critical for information extraction and knowledge graphs.
- Foundation for question answering and search systems.

### Common Entity Types

- **PERSON**: Names of people
- **ORG**: Organizations, companies, institutions
- **GPE**: Geopolitical entities (countries, cities, states)
- **DATE**: Dates and time expressions
- **MONEY**: Monetary values
- **LOCATION**: Non-GPE locations (mountains, rivers)
- **PRODUCT**: Products and objects

### NER Approaches

1. **Rule-based**: Pattern matching and dictionaries
2. **Machine Learning**: Statistical models (CRF, HMM)
3. **Deep Learning**: Neural networks (BiLSTM-CRF, Transformers)
4. **Pre-trained Models**: spaCy, BERT-based models

## spaCy for NER

### Why spaCy?

- Fast and production-ready
- Pre-trained models for multiple languages
- Easy to use API
- Supports custom entity training
- Industrial-strength NLP library

### spaCy Pipeline

```
Text → Tokenizer → Tagger → Parser → NER → Output
```

- Each component processes and adds annotations
- Can customize or disable components for performance

## Key Insights

- NER transforms unstructured text into structured data
- Accuracy varies by domain - finance NER differs from medical NER
- Pre-trained models work well for common entities
- Custom training needed for domain-specific entities
- Always validate NER results, especially for critical applications

---

# **Hands-On Labs**

## Lab 1: Basic NER with spaCy

```python
import spacy

# Load pre-trained model
nlp = spacy.load("en_core_web_sm")

text = """
Apple Inc. was founded by Steve Jobs in Cupertino, California.
The company released the iPhone on January 9, 2007.
Tim Cook became CEO in August 2011, succeeding Steve Jobs.
"""

# Process text
doc = nlp(text)

# Extract entities
print("Entities found:")
for ent in doc.ents:
    print(f"{ent.text:20} - {ent.label_:15} - {spacy.explain(ent.label_)}")
```

## Lab 2: Entity Visualization

```python
from spacy import displacy

text = """
Amazon, headquartered in Seattle, Washington, was founded by Jeff Bezos in 1994.
The company started as an online bookstore and has grown into one of the world's
largest technology companies. In 2021, Andy Jassy became the CEO.
"""

doc = nlp(text)

# Visualize entities in Jupyter or save to HTML
displacy.render(doc, style="ent", jupyter=False)
# For saving to file:
# html = displacy.render(doc, style="ent")
# with open("entities.html", "w") as f:
#     f.write(html)
```

## Lab 3: Entity Extraction and Analysis

```python
from collections import Counter

def extract_entities_by_type(text, entity_type):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ == entity_type]
    return entities

def analyze_entities(text):
    doc = nlp(text)
    
    # Count entities by type
    entity_counts = Counter([ent.label_ for ent in doc.ents])
    
    # Group entities by type
    entities_by_type = {}
    for ent in doc.ents:
        if ent.label_ not in entities_by_type:
            entities_by_type[ent.label_] = []
        entities_by_type[ent.label_].append(ent.text)
    
    return entity_counts, entities_by_type

news_text = """
Microsoft announced on Monday that it would acquire GitHub for $7.5 billion.
The deal was completed in October 2018. Satya Nadella, CEO of Microsoft,
praised GitHub's community of developers. GitHub, based in San Francisco,
was founded in 2008 by Tom Preston-Werner, Chris Wanstrath, and PJ Hyett.
"""

counts, entities = analyze_entities(news_text)

print("Entity Counts:")
for entity_type, count in counts.items():
    print(f"{entity_type}: {count}")

print("\nEntities by Type:")
for entity_type, entity_list in entities.items():
    print(f"\n{entity_type}:")
    for entity in set(entity_list):
        print(f"  - {entity}")
```

## Lab 4: Custom Entity Extraction

```python
def extract_all_people(text):
    """Extract all person names from text"""
    doc = nlp(text)
    people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    return list(set(people))  # Remove duplicates

def extract_all_organizations(text):
    """Extract all organization names from text"""
    doc = nlp(text)
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    return list(set(orgs))

def extract_all_locations(text):
    """Extract all locations from text"""
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents 
                 if ent.label_ in ["GPE", "LOC"]]
    return list(set(locations))

article = """
Elon Musk's company Tesla is building a new factory in Austin, Texas.
The factory will produce the Cybertruck and will employ thousands of workers.
SpaceX, another company founded by Musk, is based in Hawthorne, California.
Musk recently moved from California to Texas, citing better business conditions.
"""

print("People:", extract_all_people(article))
print("Organizations:", extract_all_organizations(article))
print("Locations:", extract_all_locations(article))
```

## Lab 5: NER for Information Extraction

```python
def extract_structured_info(text):
    """Extract structured information from text"""
    doc = nlp(text)
    
    info = {
        'people': [],
        'organizations': [],
        'locations': [],
        'dates': [],
        'money': []
    }
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            info['people'].append(ent.text)
        elif ent.label_ == "ORG":
            info['organizations'].append(ent.text)
        elif ent.label_ in ["GPE", "LOC"]:
            info['locations'].append(ent.text)
        elif ent.label_ == "DATE":
            info['dates'].append(ent.text)
        elif ent.label_ == "MONEY":
            info['money'].append(ent.text)
    
    # Remove duplicates
    for key in info:
        info[key] = list(set(info[key]))
    
    return info

business_news = """
Google announced on March 15, 2023, that Sundar Pichai will continue as CEO.
The company, based in Mountain View, California, reported revenue of $280 billion
for the fiscal year. Google's parent company, Alphabet Inc., also owns YouTube
and has offices in New York, London, and Tokyo.
"""

structured_data = extract_structured_info(business_news)
print("Extracted Information:")
for key, values in structured_data.items():
    print(f"\n{key.capitalize()}:")
    for value in values:
        print(f"  - {value}")
```

---

# **Practice Exercises**

1. Extract entities from news articles and categorize them
2. Build a knowledge graph from extracted entities and their relationships
3. Compare NER results from different spaCy models (small vs large)
4. Create a function to extract company-executive relationships
5. Analyze entity co-occurrence patterns in documents
