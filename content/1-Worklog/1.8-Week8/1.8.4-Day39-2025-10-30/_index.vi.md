---
title: "Ngày 39 - Nhận dạng thực thể được đặt tên (NER)"
weight: 4
chapter: false
pre: "<b> 1.8.4. </b>"
---

**Ngày:** 2025-10-30
**Trạng thái:** "Hoàn thành"  

---

# **Ghi chú bài giảng**

## Nhận dạng thực thể được đặt tên (NER)

### NER là gì?

- Xác định và phân loại các thực thể được đặt tên trong văn bản.
- Thực thể: con người, tổ chức, địa điểm, ngày tháng, sản phẩm, v.v.
- Quan trọng cho trích xuất thông tin và đồ thị tri thức.
- Nền tảng cho hệ thống trả lời câu hỏi và tìm kiếm.

### Các loại thực thể phổ biến

- **PERSON**: Tên người
- **ORG**: Tổ chức, công ty, tổ chức
- **GPE**: Thực thể địa chính trị (quốc gia, thành phố, bang)
- **DATE**: Biểu thức ngày và thời gian
- **MONEY**: Giá trị tiền tệ
- **LOCATION**: Địa điểm không phải GPE (núi, sông)
- **PRODUCT**: Sản phẩm và đối tượng

### Các cách tiếp cận NER

1. **Dựa trên quy tắc**: Khớp mẫu và từ điển
2. **Machine Learning**: Mô hình thống kê (CRF, HMM)
3. **Deep Learning**: Mạng nơ-ron (BiLSTM-CRF, Transformers)
4. **Mô hình đã huấn luyện trước**: spaCy, mô hình dựa trên BERT

## spaCy cho NER

### Tại sao chọn spaCy?

- Nhanh và sẵn sàng cho production
- Mô hình đã huấn luyện trước cho nhiều ngôn ngữ
- API dễ sử dụng
- Hỗ trợ huấn luyện thực thể tùy chỉnh
- Thư viện NLP chuẩn công nghiệp

### Quy trình spaCy

```
Văn bản → Tokenizer → Tagger → Parser → NER → Đầu ra
```

- Mỗi thành phần xử lý và thêm chú thích
- Có thể tùy chỉnh hoặc vô hiệu hóa các thành phần để tăng hiệu suất

## Những hiểu biết quan trọng

- NER chuyển đổi văn bản không có cấu trúc thành dữ liệu có cấu trúc
- Độ chính xác thay đổi theo lĩnh vực - NER tài chính khác NER y tế
- Mô hình đã huấn luyện trước hoạt động tốt cho các thực thể phổ biến
- Cần huấn luyện tùy chỉnh cho thực thể đặc thù lĩnh vực
- Luôn xác thực kết quả NER, đặc biệt cho ứng dụng quan trọng

---

# **Thực hành Lab**

## Lab 1: NER cơ bản với spaCy

```python
import spacy

# Tải mô hình đã huấn luyện trước
nlp = spacy.load("en_core_web_sm")

text = """
Apple Inc. was founded by Steve Jobs in Cupertino, California.
The company released the iPhone on January 9, 2007.
Tim Cook became CEO in August 2011, succeeding Steve Jobs.
"""

# Xử lý văn bản
doc = nlp(text)

# Trích xuất thực thể
print("Các thực thể tìm thấy:")
for ent in doc.ents:
    print(f"{ent.text:20} - {ent.label_:15} - {spacy.explain(ent.label_)}")
```

## Lab 2: Trực quan hóa thực thể

```python
from spacy import displacy

text = """
Amazon, có trụ sở tại Seattle, Washington, được thành lập bởi Jeff Bezos vào năm 1994.
Công ty bắt đầu như một cửa hàng sách trực tuyến và đã phát triển thành một trong những
công ty công nghệ lớn nhất thế giới. Năm 2021, Andy Jassy trở thành CEO.
"""

doc = nlp(text)

# Trực quan hóa thực thể trong Jupyter hoặc lưu thành HTML
displacy.render(doc, style="ent", jupyter=False)
# Để lưu vào file:
# html = displacy.render(doc, style="ent")
# with open("entities.html", "w", encoding="utf-8") as f:
#     f.write(html)
```

## Lab 3: Trích xuất và phân tích thực thể

```python
from collections import Counter

def extract_entities_by_type(text, entity_type):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ == entity_type]
    return entities

def analyze_entities(text):
    doc = nlp(text)
    
    # Đếm thực thể theo loại
    entity_counts = Counter([ent.label_ for ent in doc.ents])
    
    # Nhóm thực thể theo loại
    entities_by_type = {}
    for ent in doc.ents:
        if ent.label_ not in entities_by_type:
            entities_by_type[ent.label_] = []
        entities_by_type[ent.label_].append(ent.text)
    
    return entity_counts, entities_by_type

news_text = """
Microsoft thông báo hôm thứ Hai rằng họ sẽ mua GitHub với giá 7,5 tỷ đô la.
Giao dịch được hoàn thành vào tháng 10 năm 2018. Satya Nadella, CEO của Microsoft,
ca ngợi cộng đồng nhà phát triển của GitHub. GitHub, có trụ sở tại San Francisco,
được thành lập năm 2008 bởi Tom Preston-Werner, Chris Wanstrath và PJ Hyett.
"""

counts, entities = analyze_entities(news_text)

print("Số lượng thực thể:")
for entity_type, count in counts.items():
    print(f"{entity_type}: {count}")

print("\nThực thể theo loại:")
for entity_type, entity_list in entities.items():
    print(f"\n{entity_type}:")
    for entity in set(entity_list):
        print(f"  - {entity}")
```

## Lab 4: Trích xuất thực thể tùy chỉnh

```python
def extract_all_people(text):
    """Trích xuất tất cả tên người từ văn bản"""
    doc = nlp(text)
    people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    return list(set(people))  # Loại bỏ trùng lặp

def extract_all_organizations(text):
    """Trích xuất tất cả tên tổ chức từ văn bản"""
    doc = nlp(text)
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    return list(set(orgs))

def extract_all_locations(text):
    """Trích xuất tất cả địa điểm từ văn bản"""
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents 
                 if ent.label_ in ["GPE", "LOC"]]
    return list(set(locations))

article = """
Công ty Tesla của Elon Musk đang xây dựng một nhà máy mới ở Austin, Texas.
Nhà máy sẽ sản xuất Cybertruck và sẽ tuyển dụng hàng nghìn công nhân.
SpaceX, một công ty khác được Musk thành lập, có trụ sở tại Hawthorne, California.
Musk gần đây đã chuyển từ California sang Texas, nêu ra điều kiện kinh doanh tốt hơn.
"""

print("Người:", extract_all_people(article))
print("Tổ chức:", extract_all_organizations(article))
print("Địa điểm:", extract_all_locations(article))
```

## Lab 5: NER cho trích xuất thông tin

```python
def extract_structured_info(text):
    """Trích xuất thông tin có cấu trúc từ văn bản"""
    doc = nlp(text)
    
    info = {
        'người': [],
        'tổ_chức': [],
        'địa_điểm': [],
        'ngày_tháng': [],
        'tiền': []
    }
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            info['người'].append(ent.text)
        elif ent.label_ == "ORG":
            info['tổ_chức'].append(ent.text)
        elif ent.label_ in ["GPE", "LOC"]:
            info['địa_điểm'].append(ent.text)
        elif ent.label_ == "DATE":
            info['ngày_tháng'].append(ent.text)
        elif ent.label_ == "MONEY":
            info['tiền'].append(ent.text)
    
    # Loại bỏ trùng lặp
    for key in info:
        info[key] = list(set(info[key]))
    
    return info

business_news = """
Google thông báo vào ngày 15 tháng 3 năm 2023 rằng Sundar Pichai sẽ tiếp tục làm CEO.
Công ty, có trụ sở tại Mountain View, California, báo cáo doanh thu 280 tỷ đô la
cho năm tài chính. Công ty mẹ của Google, Alphabet Inc., cũng sở hữu YouTube
và có văn phòng tại New York, London và Tokyo.
"""

structured_data = extract_structured_info(business_news)
print("Thông tin đã trích xuất:")
for key, values in structured_data.items():
    print(f"\n{key.capitalize()}:")
    for value in values:
        print(f"  - {value}")
```

---

# **Bài tập thực hành**

1. Trích xuất thực thể từ bài báo và phân loại chúng
2. Xây dựng đồ thị tri thức từ các thực thể đã trích xuất và mối quan hệ của chúng
3. So sánh kết quả NER từ các mô hình spaCy khác nhau (nhỏ vs lớn)
4. Tạo hàm trích xuất mối quan hệ công ty-giám đốc điều hành
5. Phân tích các mẫu đồng xuất hiện thực thể trong tài liệu
