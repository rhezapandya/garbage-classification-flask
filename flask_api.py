from flask import Flask, request, jsonify
import numpy as np
import tempfile
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
model = load_model('GarbageClassification-1.h5')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_file = request.files['image']

        if 'image' not in request.files:
            return jsonify({'error': 'No image in the request'}), 400

        temp_image = tempfile.NamedTemporaryFile(delete=False)
        image_file.save(temp_image)
        temp_image.close()

        img = load_img(temp_image.name, target_size=(
            224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        class_probabilities = model.predict(img_array)
        waste_labels = {0: "cardboard", 1: "glass",
                        2: "metal", 3: "paper", 4: "plastic", 5: "trash"}

        response = ''

        predicted_class_index = np.argmax(class_probabilities)
        predicted_class_label = (
            waste_labels[predicted_class_index]).capitalize()
        predicted_class_probability = float((class_probabilities[0,
                                                                 predicted_class_index]))
        predicted_class_probability_percentage = "{:.2f}%".format(
            predicted_class_probability * 100)

        response = {'Prediction': predicted_class_label,
                    'Class': predicted_class_label,
                    'ProbabilityPercentage': predicted_class_probability_percentage}

        print(response)
        return jsonify(response)

    except Exception as e:
        print('An error occurred:', str(e))
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
