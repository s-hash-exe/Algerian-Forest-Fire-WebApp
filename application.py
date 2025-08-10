from flask import Flask, render_template, request
import pickle
import os

application = Flask(__name__)  # EB entry point

# Load models once at startup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
ridge_path = os.path.join(BASE_DIR, 'models', 'ridge.pkl')

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

with open(ridge_path, 'rb') as f:
    ridge = pickle.load(f)

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/predictData', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        try:
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
        except Exception as e:
            return render_template('home.html', result=f"Error: {str(e)}")
    else:
        return render_template('home.html', result='TBA')

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000)
