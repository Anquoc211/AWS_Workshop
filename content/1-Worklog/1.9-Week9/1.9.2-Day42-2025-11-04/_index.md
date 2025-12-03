---
title: "Day 42 - Advanced Text Classification"
weight: 2
chapter: false
pre: "<b> 1.9.2. </b>"
---

**Date:** 2025-11-04 (Tuesday)  
**Status:** "Done"  

---

# **Lecture Notes**

## Text Classification Revision

### Key Concepts from Week 8

- Text classification assigns predefined categories to documents
- Feature extraction: TF-IDF, word embeddings, contextualized representations
- Common algorithms: Naive Bayes, SVM, Logistic Regression, Neural Networks
- Evaluation metrics: accuracy, precision, recall, F1-score

### Advanced Feature Engineering

**TF-IDF Deep Dive**
- Term Frequency: how often a word appears in a document
- Inverse Document Frequency: how rare a word is across all documents
- Formula: TF-IDF = TF × log(N/DF)
- Balances common and distinctive terms

**N-gram Features**
- Unigrams: single words
- Bigrams: two consecutive words
- Trigrams: three consecutive words
- Character n-grams: useful for handling typos and morphology

## Advanced Classification Techniques

### Ensemble Methods

**Why Ensemble?**
- Combine multiple models for better performance
- Reduce overfitting through diversity
- More robust to noise and outliers

**Common Approaches**
- Voting: majority vote from multiple classifiers
- Bagging: Bootstrap Aggregating (e.g., Random Forest)
- Boosting: sequential model improvement (e.g., XGBoost, AdaBoost)
- Stacking: meta-learner combines base models

### Handling Imbalanced Data

**Problem:**
- Real-world data often has unequal class distribution
- Models biased toward majority class
- Poor performance on minority class

**Solutions:**
- Oversampling: SMOTE (Synthetic Minority Over-sampling)
- Undersampling: reduce majority class samples
- Class weights: penalize errors on minority class more
- Ensemble methods: EasyEnsemble, BalancedRandomForest

### Cross-Validation Strategies

**K-Fold Cross-Validation**
- Split data into K folds
- Train on K-1 folds, validate on remaining fold
- Repeat K times, average results
- Reduces overfitting, better performance estimate

**Stratified K-Fold**
- Maintains class distribution in each fold
- Critical for imbalanced datasets
- Ensures representative training/validation sets

## Key Insights

- Feature engineering often more important than algorithm choice
- Always baseline with simple models before complex ones
- Cross-validation essential for reliable performance estimates
- Domain knowledge crucial for feature selection
- Monitor both overall accuracy and per-class performance

---

# **Hands-On Labs**

## Lab 1: Advanced TF-IDF Features

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd

# Sample dataset
texts = [
    "Machine learning is fascinating",
    "Deep learning neural networks",
    "Natural language processing",
    "Computer vision image recognition",
    "Reinforcement learning agents",
    "Supervised learning algorithms",
    "Unsupervised clustering methods",
    "Classification and regression",
]

labels = ["ML", "DL", "NLP", "CV", "RL", "ML", "ML", "ML"]

# Advanced TF-IDF with n-grams
vectorizer = TfidfVectorizer(
    max_features=100,
    ngram_range=(1, 3),  # unigrams, bigrams, trigrams
    min_df=1,
    max_df=0.8,
    sublinear_tf=True  # apply sublinear tf scaling
)

X = vectorizer.fit_transform(texts)
print(f"Feature matrix shape: {X.shape}")
print(f"\nTop features: {vectorizer.get_feature_names_out()[:20]}")

# Analyze feature importance
feature_names = vectorizer.get_feature_names_out()
tfidf_scores = X.toarray().sum(axis=0)
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': tfidf_scores
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10))
```

## Lab 2: Ensemble Classification

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Sample data preparation
texts = [
    "excellent product highly recommend",
    "terrible waste of money",
    "good value for price",
    "worst purchase ever made",
    "amazing quality very satisfied",
    "disappointing not worth it",
    "fantastic experience love it",
    "poor quality broke quickly"
] * 10  # Repeat for more data

labels = ["positive", "negative", "positive", "negative", 
          "positive", "negative", "positive", "negative"] * 10

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# Vectorize
vectorizer = TfidfVectorizer(max_features=50)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Individual classifiers
nb_clf = MultinomialNB()
svm_clf = SVC(kernel='linear', probability=True)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Voting Ensemble
voting_clf = VotingClassifier(
    estimators=[
        ('nb', nb_clf),
        ('svm', svm_clf),
        ('rf', rf_clf)
    ],
    voting='soft'  # use probability estimates
)

# Train and evaluate each classifier
classifiers = {
    'Naive Bayes': nb_clf,
    'SVM': svm_clf,
    'Random Forest': rf_clf,
    'Voting Ensemble': voting_clf
}

for name, clf in classifiers.items():
    clf.fit(X_train_vec, y_train)
    score = clf.score(X_test_vec, y_test)
    print(f"{name} Accuracy: {score:.4f}")
```

## Lab 3: Handling Imbalanced Data

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter

# Create imbalanced dataset
texts_imbalanced = [
    "spam message win prize",
    "spam free money click",
    "spam urgent account update",
] + [
    "normal email from colleague",
    "normal meeting invitation tomorrow",
    "normal project update report",
    "normal lunch plans friday",
    "normal document attachment review",
    "normal weekly team meeting",
    "normal budget approval request",
] * 5

labels_imbalanced = ["spam"] * 3 + ["normal"] * 35

print(f"Original distribution: {Counter(labels_imbalanced)}")

# Vectorize
X_imb = vectorizer.fit_transform(texts_imbalanced)
y_imb = labels_imbalanced

# Strategy 1: Class Weights
lr_weighted = LogisticRegression(class_weight='balanced', max_iter=1000)
lr_weighted.fit(X_imb, y_imb)
print(f"\nWeighted Logistic Regression trained")

# Strategy 2: SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_imb.toarray(), y_imb)
print(f"After SMOTE: {Counter(y_resampled)}")

lr_smote = LogisticRegression(max_iter=1000)
lr_smote.fit(X_resampled, y_resampled)
print(f"SMOTE Logistic Regression trained")

# Strategy 3: Undersampling
undersample = RandomUnderSampler(random_state=42)
X_under, y_under = undersample.fit_resample(X_imb.toarray(), y_imb)
print(f"After Undersampling: {Counter(y_under)}")
```

## Lab 4: Cross-Validation for Robust Evaluation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score

# Prepare data
texts_cv = [
    "positive review great product",
    "negative review bad quality",
    "positive review excellent service",
    "negative review poor experience",
] * 20

labels_cv = ["positive", "negative", "positive", "negative"] * 20

X_cv = vectorizer.fit_transform(texts_cv)
y_cv = labels_cv

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

classifiers_cv = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear')
}

for name, clf in classifiers_cv.items():
    # Accuracy scores
    cv_scores = cross_val_score(clf, X_cv, y_cv, cv=skf, scoring='accuracy')
    
    # F1 scores
    f1_scores = cross_val_score(clf, X_cv, y_cv, cv=skf, 
                                 scoring=make_scorer(f1_score, average='weighted'))
    
    print(f"\n{name}:")
    print(f"  Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"  F1 Score: {f1_scores.mean():.4f} (+/- {f1_scores.std() * 2:.4f})")
    print(f"  Fold scores: {cv_scores}")
```

## Lab 5: Complete Advanced Classification Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Build pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(random_state=42))
])

# Parameter grid for tuning
param_grid = {
    'tfidf__max_features': [50, 100, 200],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf__min_df': [1, 2],
    'clf__n_estimators': [50, 100, 200],
    'clf__max_depth': [None, 10, 20],
    'clf__min_samples_split': [2, 5]
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

# Sample data
texts_full = texts_cv
labels_full = labels_cv

# Fit
print("Starting grid search...")
grid_search.fit(texts_full, labels_full)

# Results
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Test best model
best_model = grid_search.best_estimator_
test_texts = ["amazing product", "terrible service"]
predictions = best_model.predict(test_texts)
print(f"\nTest predictions: {predictions}")
```

---

# **Practice Exercises**

1. Build a multi-class text classifier for news categories
2. Implement a pipeline with custom feature extractors
3. Compare different resampling strategies on imbalanced data
4. Perform hyperparameter tuning with cross-validation
5. Analyze feature importance and model interpretability
