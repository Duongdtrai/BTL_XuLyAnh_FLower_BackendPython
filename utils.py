import os
import pickle

import cv2  # thư viện opencv
import matplotlib.pyplot as plt
import numpy as np

data_dir = "flowers"

categories = [
    "daisy",
    "dandelion",
    "rose",
    "sunflower",
    "tulip",
]  # danh mục các folder của loài hoa

data = []


def make_data():
    for category in categories:
        path = os.path.join(data_dir, category)  # ./../flowers/daisy
        label = categories.index(category)
        print("path: ", path)
        print("label: ", os.listdir(path))
        for img_name in os.listdir(path):
            image_path = os.path.join(path, img_name)
            image = cv2.imread(image_path)
            cv2.imshow("image sdf sđf", image)
            print("image_path: ", image_path)
            print("img_name: ", img_name)
            try:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                image = np.array(image, dtype=np.float32)
                data.append([image, label])

            except Exception as e:
                pass
        print(len(data))
        pik = open("data.pickle", "wb")
        pickle.dump(data, pik)
        pik.close

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def load_data():
    pick = open("data.pickle", "rb")
    data = pickle.load(pick)
    pick.close
    np.random.shuffle(data)
    feature = []
    labels = []
    for img, label in data:
        feature.append(img)
        labels.append(label)
    feature = np.array(feature, dtype=np.float32)
    labels = np.array(labels)
    # print("feature: ", feature)
    # print("labels: ", labels)
    feature = feature / 255.0
    return [feature, labels]


make_data()
