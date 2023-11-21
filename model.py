# Import các thư viện cần thiết
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from utils import load_data  # import hàm load_data từ file utils

# Load dữ liệu từ hàm load_data
(feature, labels) = load_data()

# Phân chia dữ liệu thành tập huấn luyện và tập kiểm tra (90% huấn luyện, 10% kiểm tra)
x_train, x_test, y_train, y_test = train_test_split(feature, labels, test_size=0.1)

# Danh sách các loại hoa trong dữ liệu
categories = [
    "daisy",
    "dandelion",
    "rose",
    "sunflower",
    "tulip",
]

# Tạo lớp đầu vào với kích thước (224, 224, 3) đại diện cho ảnh màu 224x224 pixels với 3 kênh màu (RGB)
input_layer = tf.keras.layers.Input([224, 224, 3])

# Lớp Convolutional Neural Network (CNN) với 32 bộ lọc, kích thước kernel 5x5, và hàm kích hoạt ReLU
conv1 = tf.keras.layers.Conv2D(
    filters=32, kernel_size=(5, 5), padding="Same", activation="relu"
)(input_layer)
# Lớp MaxPooling để giảm kích thước ảnh
pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

# Tương tự, thêm các lớp Convolutional, MaxPooling khác để xây dựng CNN
conv2 = tf.keras.layers.Conv2D(
    filters=64, kernel_size=(3, 3), padding="Same", activation="relu"
)(pool1)
pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

conv3 = tf.keras.layers.Conv2D(
    filters=96, kernel_size=(3, 3), padding="Same", activation="relu"
)(pool2)
pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)

conv4 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding="Same")(pool3)
pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4)

# Lớp Flatten để chuyển đổi dữ liệu thành dạng vector trước khi đưa vào lớp Fully Connected (Dense)
flt1 = tf.keras.layers.Flatten()(pool4)

# Lớp Fully Connected với 512 đơn vị và hàm kích hoạt ReLU
dn1 = tf.keras.layers.Dense(512, activation="relu")(flt1)
# Lớp đầu ra với 5 đơn vị và hàm kích hoạt softmax cho phân loại đa lớp
out = tf.keras.layers.Dense(5, activation="softmax")(dn1)

# Tạo mô hình bằng cách kết nối các lớp lại với nhau
model = tf.keras.Model(input_layer, out)

# Biên soạn mô hình với optimizer là Adam, hàm mất mát là sparse categorical crossentropy
# và đánh giá hiệu suất bằng độ chính xác
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Huấn luyện mô hình trên dữ liệu huấn luyện (x_train, y_train) với batch size là 100 và 10 epochs
model.fit(x_train, y_train, batch_size=100, epochs=10)

# Lưu mô hình đã huấn luyện vào tệp "mymodel.h5"
model.save("mymodel.h5")
