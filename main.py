from trainer import classifier
import numpy as np
import json
from flask import Flask, render_template, request

app = Flask(__name__)

model_classifier = classifier()
rf_classifier,rf_scaler = model_classifier.fit_rf_classifier()
lr_classifier, lr_scaler = model_classifier.fit_lr_classifier()

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route("/result", methods=['GET', 'POST'])
def get_result():
    data = request.args.to_dict()
    
    input_list = []
    input_list.append(int(data['sex']))
    input_list.append(int(data['pneumonia']))
    input_list.append(int(data['age'])/100)
    input_list.append(int(data['diabetes']))
    input_list.append(int(data['copd']))
    input_list.append(int(data['asthma']))
    input_list.append(int(data['inmsupr']))
    input_list.append(int(data['cardiovascular']))
    input_list.append(int(data['obesity']))
    input_list.append(int(data['renal_chronic']))
    input_list.append(int(data['tobacco']))
    
    input_list = np.array(input_list)    
    
    pred_hos = rf_classifier.predict(input_list.reshape(1, -1))
    pred_icu = lr_classifier.predict(input_list.reshape(1, -1))
    
    if (pred_hos == 0):
        hos = 'Yes'
    else:
        hos = 'No'
        
    if (pred_icu == 1):
        icu = 'Yes'
    else:
        icu = 'No'
    
    return json.dumps({'status': 'OK', 'icu': icu, 'hos': hos})

if __name__ == '__main__':
    app.run()