import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename          #security with the cleaning of user file
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image    #processing of image
from tensorflow.keras.applications.resnet50 import preprocess_input     #as numpy

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = load_model('best_model.h5')

def predict_bone_health(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224)) #change the size (pixel)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # إضافة بعد الدُفعة (batch)
    img_array /= 255.0  #transfer the pixel (0-255) to (0-1) pour more performance

    # إجراء التنبؤ
    prediction = model.predict(img_array)
    fracture_probability = prediction[0][0]  # تحويل إلى نسبة مئوية

    if fracture_probability < 0.60:
        label = 'Normal'
        confidence = (1 - fracture_probability )
    else:
        label = 'Fractured'
        confidence = fracture_probability 

    return label, round(confidence, 2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST']) #method post ??
def upload():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    filename = secure_filename(file.filename)       #cleaning of the file name 
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)     #the route of the image selected
    file.save(file_path)

    # استخدام الدالة الجديدة للتنبؤ
    result, confidence = predict_bone_health(file_path, model)

    return render_template('index.html', filename=filename, result=result, confidence=confidence)  #the result part

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True) #create the static folder
    app.run(debug=False)     #true is mean server can work when we change it and print the changes 
    #when we upload it at net we should modify to false
