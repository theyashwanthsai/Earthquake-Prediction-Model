from flask import Flask, render_template, request
from ml import pred
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

app = Flask(__name__)

with open('earthquake_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)
    
lad = 28.7041
lod = 77.1025

lak = 27.7172
lok = 85.3240

las = 34.0837
los = 74.7973

lag = 26.1158
log = 91.7086

lam = 19.0760
lom = 72.8777

lac = 13.0827
loc = 80.2707

delhi = pred(lad, lod)
kathmandu = pred(lak, lok)
srinagar = pred(las, los)
guwahati = pred(lag, log)
mumbai = pred(lam, lom)
chennai = pred(lac, loc)
thiruvananthapuram = pred(8.5241 , 76.9366)
kochi = pred(9.9312, 76.2673)

# print("The predicted magnitude of the next earthquake at Srinagar: ", srinagar)
# print("The predicted magnitude of the next earthquake at Delhi: ", delhi)
# print("The predicted magnitude of the next earthquake at Kathmandu: ", kathmandu)
# print("The predicted magnitude of the next earthquake at Guwahati: ", guwahati)
# print("The predicted magnitude of the next earthquake at Mumbai: ", mumbai)
# print("The predicted magnitude of the next earthquake at Chennai: ", chennai)

@app.route('/')
def home():
    # return render_template('index.html')
    result1 = srinagar[0]
    result2 = delhi[0]
    result3 = guwahati[0]
    result4 = kathmandu[0]
    result5 = mumbai[0]
    result6 = chennai[0]
    result7 = thiruvananthapuram[0]
    result8 = kochi[0]
    
    return render_template('index.html', result1 = result1, result2 = result2, result3 = result3, result4 = result4, result5 = result5, result6 = result6, result7 = result7, result8 = result8)


@app.route('/predict', methods=['POST'])
def predict():
    
    latitude = float(request.form['lat'])
    longitude = float(request.form['long'])
    magnitude = pred(latitude, longitude)[0]
    return render_template('index.html', magnitude=magnitude)
   

    

if __name__ == '__main__':
    app.run(debug=True)
