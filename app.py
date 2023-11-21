import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model("mymodel.h5")


# Function to preprocess the image
def preprocess_image(file):
    # Read the image from the file object using cv2.imdecode
    img_array = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.array(image, dtype=np.float32)
    image = image / 255.0
    return image.reshape(1, 224, 224, 3)


# Function to get the flower name from the model prediction
def get_flower_name(prediction):
    categories = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
    return categories[np.argmax(prediction)]


# Define the route for the home page
@app.route("/")
def home():
    return render_template("index.html")


# Define the route for flower classification
@app.route("/classify", methods=["POST"])
def classify():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    try:
        image = preprocess_image(file)
        prediction = model.predict(image)
        flower_name = get_flower_name(prediction)
        return jsonify({"flower_name": flower_name})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
