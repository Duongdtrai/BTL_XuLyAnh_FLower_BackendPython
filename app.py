# Import các thư viện cần thiết
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Load mô hình đã được huấn luyện trước đó
model = tf.keras.models.load_model("mymodel.h5")

# Hàm tiền xử lý ảnh trước khi đưa vào mô hình
def preprocess_image(file):
    # Đọc ảnh từ đối tượng file sử dụng cv2.imdecode
    # Đọc dữ liệu từ đối tượng file và chuyển thành mảng byte
    img_array = np.asarray(bytearray(file.read()), dtype=np.uint8)
    
    # Giải mã mảng byte thành ảnh sử dụng OpenCV
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # Chuyển không gian màu từ BGR (OpenCV sử dụng) sang RGB (phổ biến trong học máy)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize ảnh về kích thước 224x224, kích thước phổ biến cho các mô hình máy học
    image = cv2.resize(image, (224, 224))
    
    # Chuyển đổi ảnh thành mảng NumPy với kiểu dữ liệu float32
    image = np.array(image, dtype=np.float32)
    
    # Chuẩn hóa ảnh bằng cách chia tất cả các giá trị pixel cho 255 (đưa về khoảng [0, 1])
    image = image / 255.0

    return image.reshape(1, 224, 224, 3)

# Hàm để lấy tên loại hoa từ dự đoán của mô hình
def get_flower_name(prediction):
    categories = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
    return categories[np.argmax(prediction)]

# Định nghĩa route cho trang chủ
@app.route("/")
def home():
    return render_template("index.html")

# Định nghĩa route để phân loại hoa dựa trên ảnh được gửi lên
@app.route("/classify", methods=["POST"])
def classify():
    # Kiểm tra xem có phải có phần "file" trong request không
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    # Lấy file từ request
    file = request.files["file"]
    
    # Kiểm tra xem có file nào được chọn không
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    
    try:
        # Tiền xử lý ảnh và đưa vào mô hình để dự đoán
        image = preprocess_image(file)
        prediction = model.predict(image)
        flower_name = get_flower_name(prediction)
        
        # Trả về kết quả dưới dạng JSON
        return jsonify({"flower_name": flower_name})

    except Exception as e:
        # Trả về lỗi nếu có bất kỳ ngoại lệ nào xảy ra
        return jsonify({"error": str(e)})

# Chạy ứng dụng trên cổng 5000 và chế độ debug
if __name__ == "__main__":
    app.run(debug=True)
