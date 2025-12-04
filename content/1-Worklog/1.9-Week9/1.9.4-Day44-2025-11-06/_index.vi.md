---
title: "Ngày 44 - NER nâng cao & Huấn luyện thực thể tùy chỉnh"
weight: 4
chapter: false
pre: "<b> 1.9.4. </b>"
---

**Ngày:** 2025-11-06   
**Trạng thái:** "Hoàn thành"  

---

# **Ghi chú bài giảng**

## Ôn tập NER

### Các khái niệm chính từ Tuần 8

- Named Entity Recognition xác định và phân loại thực thể trong văn bản
- Các loại phổ biến: PERSON, ORG, GPE, DATE, MONEY, LOCATION
- Các cách tiếp cận: dựa trên quy tắc, machine learning, deep learning, mô hình đã huấn luyện trước
- spaCy cung cấp khả năng NER sẵn sàng production

### Vượt xa NER cơ bản

**Loại thực thể tùy chỉnh**
- Thực thể đặc thù lĩnh vực (mã sản phẩm, thuật ngữ y tế, tham chiếu pháp lý)
- Huấn luyện mô hình tùy chỉnh cho các lĩnh vực chuyên môn
- Mở rộng mô hình đã huấn luyện trước với các loại thực thể mới

**Trích xuất quan hệ**
- Xác định mối quan hệ giữa các thực thể
- Ví dụ: "Steve Jobs founded Apple" → (Steve Jobs, founded, Apple)
- Xây dựng đồ thị tri thức từ văn bản

## Huấn luyện NER tùy chỉnh

### Yêu cầu dữ liệu huấn luyện

**Định dạng chú thích**
```python
TRAIN_DATA = [
    ("Apple is looking at buying UK startup", {
        "entities": [(0, 5, "ORG"), (27, 29, "GPE")]
    })
]
```

**Hướng dẫn chất lượng**
- Chú thích nhất quán trên tập dữ liệu
- Đủ ví dụ cho mỗi loại thực thể (khuyến nghị 100+)
- Phân phối cân bằng của các loại thực thể
- Bao gồm ví dụ tiêu cực (văn bản không có thực thể)

### Quy trình huấn luyện

**Các bước:**
1. Chuẩn bị dữ liệu huấn luyện đã chú thích
2. Khởi tạo hoặc tải mô hình cơ sở
3. Thêm loại thực thể tùy chỉnh nếu cần
4. Huấn luyện mô hình với nhiều lần lặp
5. Đánh giá trên tập test riêng biệt
6. Tinh chỉnh siêu tham số

**Thực hành tốt nhất:**
- Sử dụng transfer learning từ mô hình đã huấn luyện trước
- Bắt đầu với tốc độ học nhỏ
- Giám sát overfitting
- Lưu checkpoints trong quá trình huấn luyện

## Kỹ thuật nâng cao

### Entity Linking

- Kết nối thực thể đã trích xuất với cơ sở tri thức (Wikipedia, Wikidata)
- Phân biệt thực thể có cùng tên
- Làm giàu thông tin thực thể với dữ liệu bên ngoài

### Nhận dạng thực thể theo ngữ cảnh

- Sử dụng ngữ cảnh xung quanh để có độ chính xác tốt hơn
- Xử lý các đề cập thực thể mơ hồ
- Xem xét thông tin cấp độ tài liệu

## Những hiểu biết quan trọng

- NER tùy chỉnh thiết yếu cho ứng dụng đặc thù lĩnh vực
- Chất lượng dữ liệu huấn luyện ảnh hưởng trực tiếp đến hiệu suất mô hình
- Mô hình đã huấn luyện trước cung cấp điểm khởi đầu tuyệt vời
- Đánh giá và huấn luyện lại liên tục duy trì độ chính xác

---

# **Thực hành Lab**

## Lab 1: Tạo dữ liệu huấn luyện tùy chỉnh

```python
import spacy
from spacy.training import Example

# Định nghĩa dữ liệu huấn luyện tùy chỉnh
TRAIN_DATA = [
    ("Apple iPhone 15 Pro giá $999", {
        "entities": [(0, 5, "CÔNG_TY"), (6, 19, "SẢN_PHẨM"), (24, 28, "TIỀN")]
    }),
    ("Samsung Galaxy S24 có giá $899", {
        "entities": [(0, 7, "CÔNG_TY"), (8, 18, "SẢN_PHẨM"), (27, 31, "TIỀN")]
    }),
    ("Google Pixel 8 có sẵn với giá $699", {
        "entities": [(0, 6, "CÔNG_TY"), (7, 14, "SẢN_PHẨM"), (31, 35, "TIỀN")]
    }),
    ("Microsoft Surface Laptop bắt đầu từ $1299", {
        "entities": [(0, 9, "CÔNG_TY"), (10, 24, "SẢN_PHẨM"), (36, 41, "TIỀN")]
    }),
    ("Tesla Model 3 giá cơ bản là $40000", {
        "entities": [(0, 5, "CÔNG_TY"), (6, 13, "SẢN_PHẨM"), (28, 34, "TIỀN")]
    })
]

def create_training_examples(nlp, train_data):
    """Chuyển đổi dữ liệu huấn luyện sang định dạng spaCy Example"""
    examples = []
    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)
    return examples

# Tải mô hình trống
nlp = spacy.blank("en")

# Tạo examples
examples = create_training_examples(nlp, TRAIN_DATA)

print(f"Đã tạo {len(examples)} ví dụ huấn luyện")
for i, example in enumerate(examples[:3]):
    print(f"\nVí dụ {i+1}:")
    print(f"Văn bản: {example.text}")
    print(f"Thực thể: {[(ent.text, ent.label_) for ent in example.reference.ents]}")
```

## Lab 2: Huấn luyện mô hình NER tùy chỉnh

```python
import random
from spacy.training import Example
from spacy.util import minibatch, compounding

def train_ner_model(nlp, train_data, n_iter=30):
    """Huấn luyện mô hình NER tùy chỉnh"""
    
    # Thêm pipeline NER nếu chưa tồn tại
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    
    # Thêm nhãn
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    
    # Vô hiệu hóa các pipeline khác trong quá trình huấn luyện
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        
        # Khởi tạo optimizer
        optimizer = nlp.begin_training()
        
        # Vòng lặp huấn luyện
        for iteration in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            
            # Tạo batches
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)
                
                # Cập nhật mô hình
                nlp.update(examples, drop=0.5, losses=losses)
            
            if iteration % 5 == 0:
                print(f"Lần lặp {iteration}, Loss: {losses['ner']:.4f}")
    
    return nlp

# Dữ liệu huấn luyện mở rộng
EXPANDED_TRAIN_DATA = TRAIN_DATA + [
    ("Amazon Echo Dot bán lẻ với giá $49", {
        "entities": [(0, 6, "CÔNG_TY"), (7, 16, "SẢN_PHẨM"), (33, 36, "TIỀN")]
    }),
    ("Sony PlayStation 5 có giá $499", {
        "entities": [(0, 4, "CÔNG_TY"), (5, 18, "SẢN_PHẨM"), (27, 31, "TIỀN")]
    }),
    ("Nintendo Switch OLED giá $349", {
        "entities": [(0, 8, "CÔNG_TY"), (9, 20, "SẢN_PHẨM"), (25, 29, "TIỀN")]
    }),
] * 5  # Lặp lại để có nhiều dữ liệu huấn luyện hơn

# Huấn luyện mô hình
nlp = spacy.blank("en")
nlp = train_ner_model(nlp, EXPANDED_TRAIN_DATA, n_iter=30)

# Kiểm tra mô hình đã huấn luyện
test_text = "MacBook Pro mới của Apple có sẵn với giá $1999"
doc = nlp(test_text)

print(f"\nVăn bản kiểm tra: {test_text}")
print("Thực thể đã trích xuất:")
for ent in doc.ents:
    print(f"  {ent.text:20} -> {ent.label_}")
```

## Lab 3: Đánh giá mô hình NER

```python
from spacy.scorer import Scorer

def evaluate_ner_model(nlp, test_data):
    """Đánh giá hiệu suất mô hình NER"""
    
    scorer = Scorer()
    examples = []
    
    for text, annotations in test_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        example.predicted = nlp(text)
        examples.append(example)
    
    # Tính điểm
    scores = scorer.score(examples)
    
    return scores

# Dữ liệu test
TEST_DATA = [
    ("HP Spectre x360 bắt đầu từ $1199", {
        "entities": [(0, 2, "CÔNG_TY"), (3, 15, "SẢN_PHẨM"), (29, 34, "TIỀN")]
    }),
    ("Dell XPS 13 có sẵn với giá $999", {
        "entities": [(0, 4, "CÔNG_TY"), (5, 11, "SẢN_PHẨM"), (28, 32, "TIỀN")]
    }),
    ("Lenovo ThinkPad giá $1299", {
        "entities": [(0, 6, "CÔNG_TY"), (7, 15, "SẢN_PHẨM"), (20, 25, "TIỀN")]
    })
]

# Đánh giá
scores = evaluate_ner_model(nlp, TEST_DATA)

print("\nĐánh giá mô hình:")
print(f"Precision: {scores['ents_p']:.4f}")
print(f"Recall: {scores['ents_r']:.4f}")
print(f"F1-Score: {scores['ents_f']:.4f}")
print(f"\nĐiểm theo loại:")
for label, metrics in scores['ents_per_type'].items():
    print(f"  {label}: P={metrics['p']:.4f}, R={metrics['r']:.4f}, F={metrics['f']:.4f}")
```

## Lab 4: Trích xuất quan hệ thực thể

```python
class RelationExtractor:
    """Trích xuất mối quan hệ giữa các thực thể"""
    
    def __init__(self, nlp):
        self.nlp = nlp
    
    def extract_relations(self, text):
        """Trích xuất bộ ba chủ ngữ-quan hệ-đối tượng"""
        doc = self.nlp(text)
        relations = []
        
        # Lấy tất cả thực thể
        entities = list(doc.ents)
        
        # Mẫu đơn giản: THỰC_THỂ1 ĐỘNG_TỪ THỰC_THỂ2
        for i, ent1 in enumerate(entities):
            for ent2 in entities[i+1:]:
                # Tìm tokens giữa các thực thể
                start = ent1.end
                end = ent2.start
                
                if end > start:
                    between_tokens = doc[start:end]
                    # Tìm động từ
                    verbs = [token.lemma_ for token in between_tokens if token.pos_ == "VERB"]
                    
                    if verbs:
                        relations.append({
                            'chủ_ngữ': ent1.text,
                            'loại_chủ_ngữ': ent1.label_,
                            'quan_hệ': ' '.join(verbs),
                            'đối_tượng': ent2.text,
                            'loại_đối_tượng': ent2.label_
                        })
        
        return relations

# Tải mô hình spaCy với phân tích dependency
nlp_full = spacy.load("en_core_web_sm")

extractor = RelationExtractor(nlp_full)

# Văn bản kiểm tra
texts = [
    "Steve Jobs thành lập Apple ở Cupertino",
    "Tim Cook trở thành CEO của Apple năm 2011",
    "Microsoft mua lại GitHub với giá $7.5 tỷ",
    "Elon Musk lãnh đạo Tesla và SpaceX"
]

print("Trích xuất quan hệ:\n")
for text in texts:
    print(f"Văn bản: {text}")
    relations = extractor.extract_relations(text)
    for rel in relations:
        print(f"  ({rel['chủ_ngữ']} [{rel['loại_chủ_ngữ']}]) "
              f"--[{rel['quan_hệ']}]--> "
              f"({rel['đối_tượng']} [{rel['loại_đối_tượng']}])")
    print()
```

## Lab 5: Hệ thống NER Production

```python
import json
from pathlib import Path

class ProductionNERSystem:
    """Hệ thống NER hoàn chỉnh cho sử dụng production"""
    
    def __init__(self, model_path=None):
        if model_path and Path(model_path).exists():
            self.nlp = spacy.load(model_path)
        else:
            self.nlp = spacy.load("en_core_web_sm")
    
    def extract_entities(self, text, return_format='dict'):
        """Trích xuất thực thể với nhiều định dạng trả về"""
        doc = self.nlp(text)
        
        if return_format == 'dict':
            return {
                'văn_bản': text,
                'thực_thể': [
                    {
                        'văn_bản': ent.text,
                        'nhãn': ent.label_,
                        'bắt_đầu': ent.start_char,
                        'kết_thúc': ent.end_char
                    }
                    for ent in doc.ents
                ]
            }
        elif return_format == 'json':
            result = {
                'văn_bản': text,
                'thực_thể': [
                    {
                        'văn_bản': ent.text,
                        'nhãn': ent.label_,
                        'bắt_đầu': ent.start_char,
                        'kết_thúc': ent.end_char
                    }
                    for ent in doc.ents
                ]
            }
            return json.dumps(result, indent=2, ensure_ascii=False)
        elif return_format == 'list':
            return [(ent.text, ent.label_, ent.start_char, ent.end_char) 
                    for ent in doc.ents]
    
    def batch_extract(self, texts):
        """Xử lý nhiều văn bản một cách hiệu quả"""
        results = []
        for doc in self.nlp.pipe(texts):
            entities = [
                {
                    'văn_bản': ent.text,
                    'nhãn': ent.label_,
                    'bắt_đầu': ent.start_char,
                    'kết_thúc': ent.end_char
                }
                for ent in doc.ents
            ]
            results.append({
                'văn_bản': doc.text,
                'thực_thể': entities
            })
        return results
    
    def save_results(self, results, output_file):
        """Lưu kết quả trích xuất vào file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

# Kiểm tra hệ thống production
ner_system = ProductionNERSystem()

# Trích xuất đơn lẻ
text = "CEO Apple Inc. Tim Cook công bố iPhone 15 mới tại sự kiện ở Cupertino, California vào ngày 12 tháng 9 năm 2023."
result = ner_system.extract_entities(text, return_format='dict')

print("Trích xuất đơn lẻ:")
print(json.dumps(result, indent=2, ensure_ascii=False))

# Trích xuất hàng loạt
texts = [
    "CEO Amazon Andy Jassy lãnh đạo công ty từ Seattle.",
    "Microsoft mua lại LinkedIn với giá $26.2 tỷ năm 2016.",
    "Google được thành lập bởi Larry Page và Sergey Brin năm 1998."
]

batch_results = ner_system.batch_extract(texts)

print("\nTrích xuất hàng loạt:")
for i, result in enumerate(batch_results):
    print(f"\nVăn bản {i+1}: {result['văn_bản']}")
    print("Thực thể:")
    for ent in result['thực_thể']:
        print(f"  - {ent['văn_bản']} ({ent['nhãn']})")
```

---

# **Bài tập thực hành**

1. Tạo mô hình NER tùy chỉnh cho một lĩnh vực cụ thể (y tế, pháp lý, tài chính)
2. Xây dựng hệ thống entity linking kết nối thực thể với Wikipedia
3. Triển khai active learning cho chú thích hiệu quả
4. Tạo giao diện web cho chú thích và kiểm tra NER
5. Xây dựng đồ thị tri thức từ các thực thể và quan hệ đã trích xuất
