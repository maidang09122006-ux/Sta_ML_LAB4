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
