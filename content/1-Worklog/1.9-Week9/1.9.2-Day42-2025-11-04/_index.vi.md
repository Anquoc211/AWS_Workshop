---
title: "Ngày 42 - Phân loại văn bản nâng cao"
weight: 2
chapter: false
pre: "<b> 1.9.2. </b>"
---

**Ngày:** 2025-11-04   
**Trạng thái:** "Hoàn thành"  

---

# **Ghi chú bài giảng**

## Ôn tập phân loại văn bản

### Các khái niệm chính từ Tuần 8

- Phân loại văn bản gán các danh mục được xác định trước cho tài liệu
- Trích xuất đặc trưng: TF-IDF, word embeddings, biểu diễn ngữ cảnh
- Thuật toán phổ biến: Naive Bayes, SVM, Logistic Regression, Neural Networks
- Số liệu đánh giá: accuracy, precision, recall, F1-score

### Kỹ thuật đặc trưng nâng cao

**TF-IDF chuyên sâu**
- Term Frequency: tần suất từ xuất hiện trong tài liệu
- Inverse Document Frequency: mức độ hiếm của từ trên tất cả tài liệu
- Công thức: TF-IDF = TF × log(N/DF)
- Cân bằng giữa các từ phổ biến và đặc trưng

**Đặc trưng N-gram**
- Unigrams: từ đơn
- Bigrams: hai từ liên tiếp
- Trigrams: ba từ liên tiếp
- Character n-grams: hữu ích cho xử lý lỗi chính tả và hình thái học

## Kỹ thuật phân loại nâng cao

### Phương pháp Ensemble

**Tại sao dùng Ensemble?**
- Kết hợp nhiều mô hình để có hiệu suất tốt hơn
- Giảm overfitting thông qua đa dạng hóa
- Mạnh mẽ hơn với nhiễu và outliers

**Các cách tiếp cận phổ biến**
- Voting: bỏ phiếu đa số từ nhiều bộ phân loại
- Bagging: Bootstrap Aggregating (ví dụ: Random Forest)
- Boosting: cải thiện mô hình tuần tự (ví dụ: XGBoost, AdaBoost)
- Stacking: meta-learner kết hợp các mô hình cơ sở

### Xử lý dữ liệu mất cân bằng

**Vấn đề:**
- Dữ liệu thực tế thường có phân phối lớp không đồng đều
- Mô hình thiên vị về lớp đa số
- Hiệu suất kém trên lớp thiểu số

**Giải pháp:**
- Oversampling: SMOTE (Synthetic Minority Over-sampling)
- Undersampling: giảm mẫu lớp đa số
- Trọng số lớp: phạt lỗi trên lớp thiểu số nhiều hơn
- Phương pháp Ensemble: EasyEnsemble, BalancedRandomForest

### Chiến lược Cross-Validation

**K-Fold Cross-Validation**
- Chia dữ liệu thành K folds
- Huấn luyện trên K-1 folds, xác thực trên fold còn lại
- Lặp lại K lần, trung bình kết quả
- Giảm overfitting, ước tính hiệu suất tốt hơn

**Stratified K-Fold**
- Duy trì phân phối lớp trong mỗi fold
- Quan trọng cho tập dữ liệu mất cân bằng
- Đảm bảo tập huấn luyện/xác thực đại diện

## Những hiểu biết quan trọng

- Kỹ thuật đặc trưng thường quan trọng hơn lựa chọn thuật toán
- Luôn baseline với mô hình đơn giản trước khi dùng mô hình phức tạp
- Cross-validation thiết yếu cho ước tính hiệu suất đáng tin cậy
- Kiến thức lĩnh vực quan trọng cho lựa chọn đặc trưng
- Giám sát cả độ chính xác tổng thể và hiệu suất từng lớp

---

# **Thực hành Lab**

## Lab 1: Đặc trưng TF-IDF nâng cao

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd

# Tập dữ liệu mẫu
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

# TF-IDF nâng cao với n-grams
vectorizer = TfidfVectorizer(
    max_features=100,
    ngram_range=(1, 3),  # unigrams, bigrams, trigrams
    min_df=1,
    max_df=0.8,
    sublinear_tf=True  # áp dụng tỷ lệ tf dưới tuyến tính
)

X = vectorizer.fit_transform(texts)
print(f"Kích thước ma trận đặc trưng: {X.shape}")
print(f"\nTop đặc trưng: {vectorizer.get_feature_names_out()[:20]}")

# Phân tích tầm quan trọng đặc trưng
feature_names = vectorizer.get_feature_names_out()
tfidf_scores = X.toarray().sum(axis=0)
feature_importance = pd.DataFrame({
    'đặc_trưng': feature_names,
    'tầm_quan_trọng': tfidf_scores
}).sort_values('tầm_quan_trọng', ascending=False)

print("\nTop 10 đặc trưng quan trọng nhất:")
print(feature_importance.head(10))
```

## Lab 2: Phân loại Ensemble

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Chuẩn bị dữ liệu mẫu
texts = [
    "sản phẩm tuyệt vời rất khuyến khích",
    "tệ lãng phí tiền",
    "giá trị tốt cho giá",
    "mua hàng tồi tệ nhất từng có",
    "chất lượng tuyệt vời rất hài lòng",
    "thất vọng không đáng giá",
    "trải nghiệm tuyệt vời thích nó",
    "chất lượng kém hỏng nhanh"
] * 10  # Lặp lại để có nhiều dữ liệu hơn

labels = ["tích_cực", "tiêu_cực", "tích_cực", "tiêu_cực", 
          "tích_cực", "tiêu_cực", "tích_cực", "tiêu_cực"] * 10

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# Vector hóa
vectorizer = TfidfVectorizer(max_features=50)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Các bộ phân loại riêng lẻ
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
    voting='soft'  # sử dụng ước tính xác suất
)

# Huấn luyện và đánh giá từng bộ phân loại
classifiers = {
    'Naive Bayes': nb_clf,
    'SVM': svm_clf,
    'Random Forest': rf_clf,
    'Voting Ensemble': voting_clf
}

for name, clf in classifiers.items():
    clf.fit(X_train_vec, y_train)
    score = clf.score(X_test_vec, y_test)
    print(f"{name} Độ chính xác: {score:.4f}")
```

## Lab 3: Xử lý dữ liệu mất cân bằng

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter

# Tạo tập dữ liệu mất cân bằng
texts_imbalanced = [
    "tin nhắn spam giành giải",
    "spam tiền miễn phí nhấp",
    "spam khẩn cấp cập nhật tài khoản",
] + [
    "email bình thường từ đồng nghiệp",
    "lời mời họp bình thường ngày mai",
    "báo cáo cập nhật dự án bình thường",
    "kế hoạch ăn trưa thứ sáu bình thường",
    "đính kèm tài liệu xem xét bình thường",
    "cuộc họp nhóm hàng tuần bình thường",
    "yêu cầu phê duyệt ngân sách bình thường",
] * 5

labels_imbalanced = ["spam"] * 3 + ["bình_thường"] * 35

print(f"Phân phối gốc: {Counter(labels_imbalanced)}")

# Vector hóa
X_imb = vectorizer.fit_transform(texts_imbalanced)
y_imb = labels_imbalanced

# Chiến lược 1: Trọng số lớp
lr_weighted = LogisticRegression(class_weight='balanced', max_iter=1000)
lr_weighted.fit(X_imb, y_imb)
print(f"\nLogistic Regression có trọng số đã huấn luyện")

# Chiến lược 2: SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_imb.toarray(), y_imb)
print(f"Sau SMOTE: {Counter(y_resampled)}")

lr_smote = LogisticRegression(max_iter=1000)
lr_smote.fit(X_resampled, y_resampled)
print(f"SMOTE Logistic Regression đã huấn luyện")

# Chiến lược 3: Undersampling
undersample = RandomUnderSampler(random_state=42)
X_under, y_under = undersample.fit_resample(X_imb.toarray(), y_imb)
print(f"Sau Undersampling: {Counter(y_under)}")
```

## Lab 4: Cross-Validation cho đánh giá mạnh mẽ

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score

# Chuẩn bị dữ liệu
texts_cv = [
    "đánh giá tích cực sản phẩm tuyệt vời",
    "đánh giá tiêu cực chất lượng tệ",
    "đánh giá tích cực dịch vụ xuất sắc",
    "đánh giá tiêu cực trải nghiệm kém",
] * 20

labels_cv = ["tích_cực", "tiêu_cực", "tích_cực", "tiêu_cực"] * 20

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
    # Điểm accuracy
    cv_scores = cross_val_score(clf, X_cv, y_cv, cv=skf, scoring='accuracy')
    
    # Điểm F1
    f1_scores = cross_val_score(clf, X_cv, y_cv, cv=skf, 
                                 scoring=make_scorer(f1_score, average='weighted'))
    
    print(f"\n{name}:")
    print(f"  Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"  F1 Score: {f1_scores.mean():.4f} (+/- {f1_scores.std() * 2:.4f})")
    print(f"  Điểm các fold: {cv_scores}")
```

## Lab 5: Quy trình phân loại nâng cao hoàn chỉnh

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Xây dựng pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(random_state=42))
])

# Lưới tham số để điều chỉnh
param_grid = {
    'tfidf__max_features': [50, 100, 200],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf__min_df': [1, 2],
    'clf__n_estimators': [50, 100, 200],
    'clf__max_depth': [None, 10, 20],
    'clf__min_samples_split': [2, 5]
}

# Grid search với cross-validation
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

# Dữ liệu mẫu
texts_full = texts_cv
labels_full = labels_cv

# Fit
print("Bắt đầu grid search...")
grid_search.fit(texts_full, labels_full)

# Kết quả
print(f"\nTham số tốt nhất: {grid_search.best_params_}")
print(f"Điểm cross-validation tốt nhất: {grid_search.best_score_:.4f}")

# Kiểm tra mô hình tốt nhất
best_model = grid_search.best_estimator_
test_texts = ["sản phẩm tuyệt vời", "dịch vụ tệ"]
predictions = best_model.predict(test_texts)
print(f"\nDự đoán kiểm tra: {predictions}")
```

---

# **Bài tập thực hành**

1. Xây dựng bộ phân loại văn bản đa lớp cho các danh mục tin tức
2. Triển khai pipeline với các bộ trích xuất đặc trưng tùy chỉnh
3. So sánh các chiến lược resampling khác nhau trên dữ liệu mất cân bằng
4. Thực hiện điều chỉnh siêu tham số với cross-validation
5. Phân tích tầm quan trọng đặc trưng và khả năng giải thích mô hình
