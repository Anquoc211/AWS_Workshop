---
title: "Day 44 - Advanced NER & Custom Entity Training"
weight: 4
chapter: false
pre: "<b> 1.9.4. </b>"
---

**Date:** 2025-11-06 (Thursday)  
**Status:** "Done"  

---

# **Lecture Notes**

## NER Revision

### Key Concepts from Week 8

- Named Entity Recognition identifies and classifies entities in text
- Common types: PERSON, ORG, GPE, DATE, MONEY, LOCATION
- Approaches: rule-based, machine learning, deep learning, pre-trained models
- spaCy provides production-ready NER capabilities

### Beyond Basic NER

**Custom Entity Types**
- Domain-specific entities (product codes, medical terms, legal references)
- Training custom models for specialized domains
- Extending pre-trained models with new entity types

**Relation Extraction**
- Identifying relationships between entities
- Example: "Steve Jobs founded Apple" → (Steve Jobs, founded, Apple)
- Building knowledge graphs from text

## Custom NER Training

### Training Data Requirements

**Annotation Format**
```python
TRAIN_DATA = [
    ("Apple is looking at buying UK startup", {
        "entities": [(0, 5, "ORG"), (27, 29, "GPE")]
    })
]
```

**Quality Guidelines**
- Consistent annotation across dataset
- Sufficient examples per entity type (100+ recommended)
- Balanced distribution of entity types
- Include negative examples (text without entities)

### Training Process

**Steps:**
1. Prepare annotated training data
2. Initialize or load base model
3. Add custom entity types if needed
4. Train model with multiple iterations
5. Evaluate on held-out test set
6. Fine-tune hyperparameters

**Best Practices:**
- Use transfer learning from pre-trained models
- Start with small learning rate
- Monitor for overfitting
- Save checkpoints during training

## Advanced Techniques

### Entity Linking

- Connect extracted entities to knowledge bases (Wikipedia, Wikidata)
- Disambiguate entities with same name
- Enrich entity information with external data

### Contextual Entity Recognition

- Use surrounding context for better accuracy
- Handle ambiguous entity mentions
- Consider document-level information

## Key Insights

- Custom NER essential for domain-specific applications
- Quality of training data directly impacts model performance
- Pre-trained models provide excellent starting point
- Continuous evaluation and retraining maintain accuracy

---

# **Hands-On Labs**

## Lab 1: Creating Custom Training Data

```python
import spacy
from spacy.training import Example

# Define custom training data
TRAIN_DATA = [
    ("Apple iPhone 15 Pro costs $999", {
        "entities": [(0, 5, "COMPANY"), (6, 19, "PRODUCT"), (26, 30, "MONEY")]
    }),
    ("Samsung Galaxy S24 is priced at $899", {
        "entities": [(0, 7, "COMPANY"), (8, 18, "PRODUCT"), (33, 37, "MONEY")]
    }),
    ("Google Pixel 8 available for $699", {
        "entities": [(0, 6, "COMPANY"), (7, 14, "PRODUCT"), (29, 33, "MONEY")]
    }),
    ("Microsoft Surface Laptop starts at $1299", {
        "entities": [(0, 9, "COMPANY"), (10, 24, "PRODUCT"), (35, 40, "MONEY")]
    }),
    ("Tesla Model 3 base price is $40000", {
        "entities": [(0, 5, "COMPANY"), (6, 13, "PRODUCT"), (28, 34, "MONEY")]
    })
]

def create_training_examples(nlp, train_data):
    """Convert training data to spaCy Example format"""
    examples = []
    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)
    return examples

# Load blank model
nlp = spacy.blank("en")

# Create examples
examples = create_training_examples(nlp, TRAIN_DATA)

print(f"Created {len(examples)} training examples")
for i, example in enumerate(examples[:3]):
    print(f"\nExample {i+1}:")
    print(f"Text: {example.text}")
    print(f"Entities: {[(ent.text, ent.label_) for ent in example.reference.ents]}")
```

## Lab 2: Training Custom NER Model

```python
import random
from spacy.training import Example
from spacy.util import minibatch, compounding

def train_ner_model(nlp, train_data, n_iter=30):
    """Train custom NER model"""
    
    # Add NER pipeline if it doesn't exist
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    
    # Add labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    
    # Disable other pipelines during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        
        # Initialize optimizer
        optimizer = nlp.begin_training()
        
        # Training loop
        for iteration in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            
            # Create batches
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)
                
                # Update model
                nlp.update(examples, drop=0.5, losses=losses)
            
            if iteration % 5 == 0:
                print(f"Iteration {iteration}, Loss: {losses['ner']:.4f}")
    
    return nlp

# Expanded training data
EXPANDED_TRAIN_DATA = TRAIN_DATA + [
    ("Amazon Echo Dot retails for $49", {
        "entities": [(0, 6, "COMPANY"), (7, 16, "PRODUCT"), (29, 32, "MONEY")]
    }),
    ("Sony PlayStation 5 priced at $499", {
        "entities": [(0, 4, "COMPANY"), (5, 18, "PRODUCT"), (29, 33, "MONEY")]
    }),
    ("Nintendo Switch OLED costs $349", {
        "entities": [(0, 8, "COMPANY"), (9, 20, "PRODUCT"), (27, 31, "MONEY")]
    }),
] * 5  # Repeat for more training data

# Train model
nlp = spacy.blank("en")
nlp = train_ner_model(nlp, EXPANDED_TRAIN_DATA, n_iter=30)

# Test trained model
test_text = "The new Apple MacBook Pro is available for $1999"
doc = nlp(test_text)

print(f"\nTest Text: {test_text}")
print("Extracted Entities:")
for ent in doc.ents:
    print(f"  {ent.text:20} -> {ent.label_}")
```

## Lab 3: Evaluating NER Model

```python
from spacy.scorer import Scorer

def evaluate_ner_model(nlp, test_data):
    """Evaluate NER model performance"""
    
    scorer = Scorer()
    examples = []
    
    for text, annotations in test_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        example.predicted = nlp(text)
        examples.append(example)
    
    # Calculate scores
    scores = scorer.score(examples)
    
    return scores

# Test data
TEST_DATA = [
    ("HP Spectre x360 starts at $1199", {
        "entities": [(0, 2, "COMPANY"), (3, 15, "PRODUCT"), (26, 31, "MONEY")]
    }),
    ("Dell XPS 13 available for $999", {
        "entities": [(0, 4, "COMPANY"), (5, 11, "PRODUCT"), (26, 30, "MONEY")]
    }),
    ("Lenovo ThinkPad costs $1299", {
        "entities": [(0, 6, "COMPANY"), (7, 15, "PRODUCT"), (22, 27, "MONEY")]
    })
]

# Evaluate
scores = evaluate_ner_model(nlp, TEST_DATA)

print("\nModel Evaluation:")
print(f"Precision: {scores['ents_p']:.4f}")
print(f"Recall: {scores['ents_r']:.4f}")
print(f"F1-Score: {scores['ents_f']:.4f}")
print(f"\nPer-type scores:")
for label, metrics in scores['ents_per_type'].items():
    print(f"  {label}: P={metrics['p']:.4f}, R={metrics['r']:.4f}, F={metrics['f']:.4f}")
```

## Lab 4: Entity Relation Extraction

```python
class RelationExtractor:
    """Extract relationships between entities"""
    
    def __init__(self, nlp):
        self.nlp = nlp
    
    def extract_relations(self, text):
        """Extract subject-relation-object triples"""
        doc = self.nlp(text)
        relations = []
        
        # Get all entities
        entities = list(doc.ents)
        
        # Simple pattern: ENTITY1 VERB ENTITY2
        for i, ent1 in enumerate(entities):
            for ent2 in entities[i+1:]:
                # Find tokens between entities
                start = ent1.end
                end = ent2.start
                
                if end > start:
                    between_tokens = doc[start:end]
                    # Look for verbs
                    verbs = [token.lemma_ for token in between_tokens if token.pos_ == "VERB"]
                    
                    if verbs:
                        relations.append({
                            'subject': ent1.text,
                            'subject_type': ent1.label_,
                            'relation': ' '.join(verbs),
                            'object': ent2.text,
                            'object_type': ent2.label_
                        })
        
        return relations

# Load spaCy model with dependency parsing
nlp_full = spacy.load("en_core_web_sm")

extractor = RelationExtractor(nlp_full)

# Test texts
texts = [
    "Steve Jobs founded Apple in Cupertino",
    "Tim Cook became CEO of Apple in 2011",
    "Microsoft acquired GitHub for $7.5 billion",
    "Elon Musk leads Tesla and SpaceX"
]

print("Relation Extraction:\n")
for text in texts:
    print(f"Text: {text}")
    relations = extractor.extract_relations(text)
    for rel in relations:
        print(f"  ({rel['subject']} [{rel['subject_type']}]) "
              f"--[{rel['relation']}]--> "
              f"({rel['object']} [{rel['object_type']}])")
    print()
```

## Lab 5: Production NER Pipeline

```python
import json
from pathlib import Path

class ProductionNERSystem:
    """Complete NER system for production use"""
    
    def __init__(self, model_path=None):
        if model_path and Path(model_path).exists():
            self.nlp = spacy.load(model_path)
        else:
            self.nlp = spacy.load("en_core_web_sm")
    
    def extract_entities(self, text, return_format='dict'):
        """Extract entities with multiple return formats"""
        doc = self.nlp(text)
        
        if return_format == 'dict':
            return {
                'text': text,
                'entities': [
                    {
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    }
                    for ent in doc.ents
                ]
            }
        elif return_format == 'json':
            result = {
                'text': text,
                'entities': [
                    {
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    }
                    for ent in doc.ents
                ]
            }
            return json.dumps(result, indent=2)
        elif return_format == 'list':
            return [(ent.text, ent.label_, ent.start_char, ent.end_char) 
                    for ent in doc.ents]
    
    def batch_extract(self, texts):
        """Process multiple texts efficiently"""
        results = []
        for doc in self.nlp.pipe(texts):
            entities = [
                {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                }
                for ent in doc.ents
            ]
            results.append({
                'text': doc.text,
                'entities': entities
            })
        return results
    
    def save_results(self, results, output_file):
        """Save extraction results to file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

# Test production system
ner_system = ProductionNERSystem()

# Single extraction
text = "Apple Inc. CEO Tim Cook announced the new iPhone 15 at an event in Cupertino, California on September 12, 2023."
result = ner_system.extract_entities(text, return_format='dict')

print("Single Extraction:")
print(json.dumps(result, indent=2))

# Batch extraction
texts = [
    "Amazon CEO Andy Jassy leads the company from Seattle.",
    "Microsoft acquired LinkedIn for $26.2 billion in 2016.",
    "Google was founded by Larry Page and Sergey Brin in 1998."
]

batch_results = ner_system.batch_extract(texts)

print("\nBatch Extraction:")
for i, result in enumerate(batch_results):
    print(f"\nText {i+1}: {result['text']}")
    print("Entities:")
    for ent in result['entities']:
        print(f"  - {ent['text']} ({ent['label']})")
```

---

# **Practice Exercises**

1. Create a custom NER model for a specific domain (medical, legal, finance)
2. Build an entity linking system connecting entities to Wikipedia
3. Implement active learning for efficient annotation
4. Create a web interface for NER annotation and testing
5. Build a knowledge graph from extracted entities and relations
