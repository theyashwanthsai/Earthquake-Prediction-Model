import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Load earthquake dataset
df = pd.read_csv('database.csv')

# Preprocess data by selecting relevant features and cleaning data
df = df[['Latitude', 'Longitude', 'Magnitude']].dropna()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['Latitude', 'Longitude']], df['Magnitude'], test_size=0.2, random_state=42)

# Train Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save trained model to file
with open('earthquake_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)

# Load saved model from file
with open('earthquake_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)




def pred(latitude, longitude):
    return rf_model.predict([[latitude, longitude]])
# Make prediction for a specific location

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



print("The predicted magnitude of the next earthquake at Srinagar: ", srinagar)
print("The predicted magnitude of the next earthquake at Delhi: ", delhi)
print("The predicted magnitude of the next earthquake at Kathmandu: ", kathmandu)
print("The predicted magnitude of the next earthquake at Guwahati: ", guwahati)
print("The predicted magnitude of the next earthquake at Mumbai: ", mumbai)
print("The predicted magnitude of the next earthquake at Chennai: ", chennai)


# accuracy = 0.9241777017858995
# best_fit.score(X_test, y_test)
# 0.8749008584467053