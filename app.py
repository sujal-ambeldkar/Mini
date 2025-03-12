import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, session, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import secrets
import gdown

app = Flask(__name__)

app.secret_key = secrets.token_hex(16)

model = load_model('plant_disease_modelfinal.h5')
class_labels = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

class_info = {
    "Bacterial Spot": {
        "description": "Bacterial spot causes dark, water-soaked lesions on leaves and fruits, which later turn brown and necrotic.",
        "cause": "Caused by Xanthomonas bacteria, often spread by water splashes, infected seeds, or contaminated tools.",
        "solution": "Use copper-based fungicides, avoid overhead watering, and practice crop rotation."
    },
    "Healthy_Bell_Pepper": {
        "description": "No visible signs of disease. The plant is in good health with normal leaf coloration and growth.",
        "cause": "Proper nutrition, watering, and disease-free environment.",
        "solution": "Continue regular care, monitor for early disease signs, and maintain good soil health."
    },
    "Potato___Early_blight": {
        "description": "Early blight causes brown spots with concentric rings on older leaves, leading to yellowing and defoliation.",
        "cause": "Caused by the fungus Alternaria solani, thrives in warm, humid conditions.",
        "solution": "Use resistant varieties, apply fungicides, and practice proper plant spacing to reduce humidity."
    },
    "Potato___Late_blight": {
        "description": "Late blight causes dark, water-soaked lesions on leaves and stems, rapidly spreading and leading to plant collapse.",
        "cause": "Caused by the pathogen Phytophthora infestans, which spreads rapidly in cool, wet conditions.",
        "solution": "Apply fungicides, remove infected plants immediately, and ensure good air circulation."
    },
    "Potato___healthy": {
        "description": "No disease symptoms detected. The plant is growing well.",
        "cause": "Adequate watering, nutrient-rich soil, and disease prevention strategies.",
        "solution": "Maintain good agricultural practices and monitor regularly."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "description": "Tomato Yellow Leaf Curl Virus causes curled, yellowing leaves and stunted plant growth.",
        "cause": "Transmitted by whiteflies, affecting plant development and reducing yield.",
        "solution": "Control whiteflies using neem oil or insecticides, and plant resistant tomato varieties."
    },
    "Tomato__Tomato_mosaic_virus": {
        "description": "Tomato Mosaic Virus causes mottled, wrinkled leaves and poor fruit development.",
        "cause": "Spread through contaminated hands, tools, and infected plant material.",
        "solution": "Disinfect tools, avoid handling plants excessively, and remove infected plants immediately."
    },
    "Tomato_healthy": {
        "description": "The tomato plant appears healthy with no visible disease symptoms.",
        "cause": "Proper watering, fertilization, and pest management.",
        "solution": "Continue maintaining healthy growth conditions and monitor for any potential issues."
    }
}

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)

    predicted_label = list(class_info.keys())[predicted_index] if predicted_index < len(class_info) else "Unknown"
    disease_info = class_info.get(predicted_label, {
        "description": "No description available.",
        "cause": "Cause not available.",
        "solution": "Solution not available."
    })

    return predicted_label, disease_info

@app.route("/", methods=["GET", "POST"])
def upload_file():
    """Handles file upload and prediction."""
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Get prediction
            predicted_label, disease_info = predict_image(filepath)

            # Save in session history
            if "history" not in session:
                session["history"] = []
            
            session["history"].append({
                "filename": filename,
                "result": predicted_label,
                "description": disease_info["description"],
                "cause": disease_info["cause"],
                "solution": disease_info["solution"]
            })
            session.modified = True  # Save session changes
            print("Session History:", session["history"])

            return render_template(
                "result.html",
                filename=filename,
                result=predicted_label,
                description=disease_info["description"],
                cause=disease_info["cause"],
                solution=disease_info["solution"],
                history=session.get("history", [])
            )
    return render_template("indexdeep.html")
@app.route('/weed.html')
def about():
    return render_template('weed.html')

@app.route('/common.html')
def common():
    return render_template('common.html')

@app.route('/fungal.html')
def fungal():
    return render_template('fungal.html')


if __name__ == "__main__":
    app.run(debug=True)