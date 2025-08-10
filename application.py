from flask import Flask, render_template, request, jsonify
import pickle

application = Flask(__name__)
app = application

scaler = None 
ridge = None

@app.before_request
def load_models():
    global scaler, ridge 
    scaler = pickle.load(open('./models/scaler.pkl', 'rb'))
    ridge = pickle.load(open('./models/ridge.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictData', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        data_scaled = scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        data_pred = ridge.predict(data_scaled)
        return render_template('home.html', result=round(data_pred[0], 2))

    else:
        return render_template('home.html', result='TBA')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)