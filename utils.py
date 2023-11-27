import os
import pickle
import cv2  # Thư viện OpenCV
import matplotlib.pyplot as plt
import numpy as np

data_dir = "flowers"  # Thư mục chứa các thư mục con cho từng loại hoa

# Danh sách các loại hoa
categories = [
    "daisy",
    "dandelion",
    "rose",
    "sunflower",
    "tulip",
]

data = []  # Danh sách để lưu dữ liệu hình ảnh và nhãn tương ứng

def make_data():
    for category in categories:
        path = os.path.join(data_dir, category)  # Đường dẫn đến thư mục của mỗi loại hoa
        label = categories.index(category)  # Gán một nhãn duy nhất cho mỗi loại hoa

        # Duyệt qua các hình ảnh trong loại hoa hiện tại
        for img_name in os.listdir(path):
            image_path = os.path.join(path, img_name)  # Đường dẫn đầy đủ đến hình ảnh hiện tại
            image = cv2.imread(image_path)  # Đọc hình ảnh bằng thư viện OpenCV

            cv2.imshow("image sdf sđf", image)  # Hiển thị hình ảnh (để kiểm tra lỗi)

            # In thông tin về hình ảnh hiện tại
            print("image_path: ", image_path)
            print("img_name: ", img_name)

            try:
                # Tiền xử lý hình ảnh
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển đổi từ BGR sang RGB
                image = cv2.resize(image, (224, 224))  # Thay đổi kích thước hình ảnh thành (224, 224)
                image = np.array(image, dtype=np.float32)  # Chuyển hình ảnh thành mảng NumPy kiểu float32

                # Thêm hình ảnh và nhãn vào danh sách dữ liệu
                data.append([image, label])

            except Exception as e:
                # Nếu có lỗi (ví dụ, không thể tải hình ảnh), bỏ qua hình ảnh hiện tại
                pass

        print(len(data))

        # Lưu danh sách dữ liệu vào một tệp pickle sau khi xử lý mỗi loại hoa
        pik = open("data.pickle", "wb")
        pickle.dump(data, pik)
        pik.close

# Hàm để tải dữ liệu từ tệp pickle đã lưu
def load_data():
    pick = open("data.pickle", "rb")
    data = pickle.load(pick)
    pick.close
    np.random.shuffle(data)  # Xáo trộn dữ liệu ngẫu nhiên

    # Tách các đặc trưng và nhãn từ dữ liệu đã tải
    feature = []
    labels = []
    for img, label in data:
        feature.append(img)
        labels.append(label)
    
    feature = np.array(feature, dtype=np.float32)
    labels = np.array(labels)
    feature = feature / 255.0  # Chuẩn hóa giá trị pixel về khoảng từ 0 đến 1

    return [feature, labels]

# Gọi hàm make_data để xử lý và lưu dữ liệu
make_data()
