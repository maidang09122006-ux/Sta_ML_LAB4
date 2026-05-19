# LAB4 - Decision Tree and Random Forest from Scratch

## 1. Mô tả bài lab

Bài lab này triển khai mô hình Decision Tree và Random Forest bằng NumPy để phân loại chất lượng rượu dựa trên bộ dữ liệu Wine Quality.

Sau đó, kết quả của mô hình tự cài đặt được so sánh với mô hình tương ứng trong thư viện scikit-learn thông qua các chỉ số:

- Accuracy
- Precision
- Recall
- F1-score

## 2. Dataset

Dataset sử dụng gồm 2 file:

- `winequality-red.csv`: dữ liệu rượu vang đỏ
- `winequality-white.csv`: dữ liệu rượu vang trắng

Hai bộ dữ liệu được gộp lại thành một dataset chung.

Cột `wine_type` được thêm vào để phân biệt:

- `0`: red wine
- `1`: white wine

Bài toán được chuyển thành phân loại nhị phân:

- `quality >= 6`: good wine, nhãn `1`
- `quality < 6`: bad wine, nhãn `0`

## 3. Cấu trúc thư mục

```text
LAB4/
│
├── data/
│   ├── winequality-red.csv
│   ├── winequality-white.csv
│   └── winequality.names
│
├── results/
│   └── metrics.csv
│
├── assignment.ipynb
├── decision_tree.py
├── random_forest.py
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```

## 4. Mô tả các file chính

### `decision_tree.py`

File này chứa class `DecisionTreeClassifier` được xây dựng bằng NumPy.

Các chức năng chính:

- Tính Gini Impurity
- Tìm điều kiện chia tốt nhất
- Xây dựng cây quyết định bằng đệ quy
- Dự đoán nhãn cho dữ liệu mới
- Hỗ trợ tham số `max_features` để dùng trong Random Forest

### `random_forest.py`

File này chứa class `RandomForestClassifier` được xây dựng dựa trên nhiều Decision Tree.

Các chức năng chính:

- Bootstrap sampling
- Huấn luyện nhiều cây Decision Tree
- Chọn ngẫu nhiên một phần feature tại mỗi lần split
- Tổng hợp kết quả bằng majority voting

### `main.py`

File chạy chính của bài lab.

Các bước chính:

1. Load dữ liệu red wine và white wine
2. Gộp hai bộ dữ liệu
3. Chuyển bài toán thành binary classification
4. Chia train/test
5. Train Decision Tree tự cài đặt bằng NumPy
6. Train Random Forest tự cài đặt bằng NumPy
7. Train Decision Tree bằng scikit-learn
8. Train Random Forest bằng scikit-learn
9. So sánh kết quả bằng Accuracy, Precision, Recall và F1-score
10. Lưu kết quả vào `results/metrics.csv`

## 5. Cài đặt thư viện

Chạy lệnh:

```bash
pip install -r requirements.txt
```

## 6. Cách chạy chương trình

Chạy file chính:

```bash
python main.py
```

Sau khi chạy, chương trình sẽ in kết quả ra terminal và lưu kết quả vào:

```text
results/metrics.csv
```

## 7. Tham số mô hình

Các tham số chính được sử dụng:

```python
max_depth = 5
min_samples_split = 10
min_samples_leaf = 5
n_estimators = 100
max_features = "sqrt"
```

Ý nghĩa:

- `max_depth=5`: giới hạn độ sâu của cây để giảm overfitting.
- `min_samples_split=10`: một node cần ít nhất 10 mẫu mới được tách tiếp.
- `min_samples_leaf=5`: mỗi leaf node cần có ít nhất 5 mẫu.
- `n_estimators=100`: Random Forest sử dụng 100 cây.
- `max_features="sqrt"`: tại mỗi lần split, chỉ chọn ngẫu nhiên một phần feature để xét.

Các tham số này được chọn để cân bằng giữa hiệu quả mô hình, khả năng tránh overfitting và thời gian chạy.

## 8. Kết quả thực nghiệm

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Decision Tree NumPy | 0.7362 | 0.7871 | 0.7995 | 0.7932 |
| Random Forest NumPy | 0.7546 | 0.7745 | 0.8639 | 0.8168 |
| Decision Tree sklearn | 0.7354 | 0.7868 | 0.7983 | 0.7925 |
| Random Forest sklearn | 0.7585 | 0.7781 | 0.8651 | 0.8193 |

## 9. Nhận xét

Kết quả cho thấy Random Forest đạt hiệu quả tốt hơn Decision Tree trên bài toán phân loại chất lượng rượu.

Decision Tree tự cài đặt bằng NumPy có F1-score gần tương đương với Decision Tree của scikit-learn.

Random Forest tự cài đặt cũng đạt kết quả khá gần với Random Forest của scikit-learn. Mô hình Random Forest của scikit-learn nhỉnh hơn một chút do thư viện đã được tối ưu tốt hơn về thuật toán và hiệu năng.

## 10. Kết luận

Bài lab đã hoàn thành các yêu cầu chính:

- Tự xây dựng Decision Tree bằng NumPy
- Tự xây dựng Random Forest bằng NumPy
- So sánh với mô hình Decision Tree và Random Forest của scikit-learn
- Đánh giá mô hình bằng Accuracy, Precision, Recall và F1-score
- Lưu kết quả thực nghiệm ra file `results/metrics.csv`
