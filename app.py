from flask import send_from_directory, Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_path = 'classifier.h5'
classifier = tf.keras.models.load_model(model_path)

def preprocess_image(img_path):
    if not os.path.exists(img_path):
        raise ValueError("Image file does not exist.")

    
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64, 64))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0  

    return img_array

def predict(img_path, model):
    image_array = preprocess_image(img_path)
    prediction = model.predict(image_array)
    
    if prediction[0][0] < 0.5:
        return 'The uploded image is Female'
    else:
        return 'The uploaded image is Male'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        predicted_class = predict(filepath, classifier)

        image_url = url_for('uploaded_file', filename=filename)

        return render_template('result.html', predicted_class=predicted_class, image_url=image_url)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
