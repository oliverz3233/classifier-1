from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
from processing.mfe.dsp import generate_features
import wave

app = Flask(__name__)
CORS(app)   

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    audio_file = request.files['audio']
    aud_stream = audio_file.read()
    with open("audio.wav", "wb") as aud:
        aud.write(aud_stream)
    '''with wave.open('audio.wav', 'rb') as wav_file:
        print("Get header information")
        print(wav_file.getnchannels())
        print(wav_file.getsampwidth())
        print(wav_file.getframerate())
        print(wav_file.getnframes())
        print(wav_file.getcomptype())
        print(wav_file.getcompname())'''
    input_filename = 'audio.wav'
    samrate, data = wavfile.read(input_filename)
    listData = []
    if(len(data.shape) == 1):
        for i in range(0, data.shape[0]):
            listData.append(data[i])
    else:
        for i in range(0, data.shape[0]):
            listData.append(data[i][0])
    raw_features = np.array(listData)
    print(listData[len(listData)-1])
    result = generate_features(implementation_version=3,
                            draw_graphs=False,
                            raw_data=raw_features,
                            axes=[""],
                            sampling_freq=samrate,
                            frame_length=0.02,
                            frame_stride=0.01,
                            num_filters=40,
                            fft_length=256,
                            low_frequency=0,
                            high_frequency=0,
                            win_size=1000,
                            noise_floor_db=-52)
    response = {'Result': result}
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='localhost', port=5000)