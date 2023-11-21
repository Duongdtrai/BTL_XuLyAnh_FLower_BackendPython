import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Import hàm load_data từ module utils
from utils import load_data

# Sử dụng hàm load_data() để tải dữ liệu (giả sử dữ liệu hình ảnh của các loại hoa)
(feature, labels) = load_data()

# Phân chia dữ liệu thành tập huấn luyện và tập kiểm tra
(x_train, x_test, y_train, y_test) = train_test_split(feature, labels, test_size=0.1)

# Định nghĩa các tên loại hoa
categories = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# Tải mô hình đã được lưu trữ trước đó từ tệp tin "./mymodel.h5"
model = tf.keras.models.load_model("./mymodel.h5")

# Đánh giá mô hình trên tập kiểm tra và hiển thị kết quả
model.evaluate(x_test, y_test, verbose=1)

# Dự đoán các hình ảnh từ tập kiểm tra sử dụng mô hình
prediction = model.predict(x_test)

# Tạo và hiển thị các hình ảnh trong một biểu đồ
plt.figure(figsize=(100, 100))

for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.imshow(x_test[i])  # Hiển thị hình ảnh
    plt.xlabel(
        "Actual: "
        + categories[y_test[i]]
        + "\n"
        + "Predicted: "
        + categories[np.argmax(prediction[i])]
    )  # Hiển thị nhãn thực tế và nhãn được dự đoán

    plt.xticks([])  # Ẩn các chỉ số trục x

plt.show()  # Hiển thị biểu đồ
