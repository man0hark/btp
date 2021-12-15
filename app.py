import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template
from io import BytesIO
import base64
from scipy.io import wavfile
import python_speech_features
from matplotlib.figure import Figure
from tensorflow.keras.models import load_model
from flask import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


app = Flask(__name__) 
@app.route('/')
def home():
    return render_template('Home.html')
@app.route('/success',methods=['POST'])
def success():
    f = request.files['file']
    sr, data = wavfile.read(f)
    mfcc_speech = python_speech_features.mfcc(signal=data, samplerate=sr)
    model1 = load_model('my_model.h5')
    ms = np.array(mfcc_speech)
    ms = ms.reshape(ms.shape[0],ms.shape[1],1)
    y_pred = model1.predict(ms)
    y_pred = y_pred.round()
    img = BytesIO()
    plt.style.use('seaborn-muted')
    plt.figure(figsize=(10,5))
    plt.subplot(2,1,1)
    plt.plot(data)
    plt.subplot(2,1,2)
    plt.plot(y_pred.round())
    plt.savefig(img, format='png',transparent=True)
    plt.axis('off')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return render_template('plot.html', plot_url=plot_url)
if __name__ == '__main__':  
    app.run(debug = True)
