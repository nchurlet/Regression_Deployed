import streamlit as st
import pickle
import joblib
import pandas as pd
from PIL import Image

# modèles:
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from keras.models import load_model


# Sidebar - this sidebar allows the user to set the parameters that will be used by the model to create the prediction.
st.sidebar.header('Renseignez les caractéristiques des biens du quartier pour obtenir une évaluation des prix des biens')


modelDecisionTreeRegressor_0 = pickle.load( open( "modelDecisionTreeRegressor_0.p", "rb" ))
modelGradientBoostingRegressor_0 = pickle.load( open( "modelGradientBoostingRegressor_0.p", "rb" ))
modelLinearRegression_0 = pickle.load( open( "modelLinearRegression0.p", "rb" ))
#modelRandomForestRegressor_0 = pickle.load( open( "modelRandomForestRegressor0.p", "rb" ))
scaler = pickle.load( open("scaler.p", "rb" ))
#model_ann_0 = pickle.load( open("model_ann_0.p", "rb" ))

# load the model
model_ann_0 = load_model('./modelann.h5')

#data_train_example = pickle.load( open( "data_train_example.p", "rb" )) 
data_train_example = joblib.load( open( "./data_train_example.p", "rb" ))# 



MedInc = st.sidebar.slider(
	"Revenus moyens des habitants du quartier", 
	min_value=float(data_train_example.min()[0]), 
	max_value=float(data_train_example.max()[0])
	)
HouseAge = st.sidebar.slider(
	"Age moyen des maisons dans le quartier",
	min_value=float(data_train_example.min()[1]),
	max_value=float(data_train_example.max()[1])
	)
AveRooms = st.sidebar.slider(
	"Nombre moyen de pièces",
	min_value=float(data_train_example.min()[2]),
	max_value=float(data_train_example.max()[2])
	)
AveBedrms = st.sidebar.slider(
	"Nombre moyen de chambres",
	min_value=float(data_train_example.min()[3]),
	max_value=float(data_train_example.max()[3])
	)
Population = st.sidebar.slider(
	"Population du quartier",
	min_value=float(data_train_example.min()[4]),
	max_value=float(data_train_example.max()[4])
	)
AveOccup = st.sidebar.slider(
	"Moyenne du nombre d'occupant",
	min_value=float(data_train_example.min()[5]),
	max_value=float(data_train_example.max()[5])
	)
Latitude = st.sidebar.slider(
	"Latitude en Californie",
	min_value=float(data_train_example.min()[6]),
	max_value=float(data_train_example.max()[6])
	)
Longitude = st.sidebar.slider(
	"Longitude en Californie",
	min_value=float(data_train_example.min()[7]),
	max_value=float(data_train_example.max()[7])
	)
# mettre une image
#![Some California Houses]("C:/Users/Utilisateur/Documents/_Simplon/A_Rendre/Regression"?auto=compress&cs=tinysrgb&h=750&w=1260)
# image = Image.open('./images/ccalifornia_house_2.PNG')
# st.image(image, use_column_width=True)
def get_data():
	data = {'Longitude': Longitude,
	'Latitude': Latitude,
	'Housing Median Age': HouseAge,
	'Total Rooms': AveRooms,
	'Total Bedrooms': AveBedrms,
	'Population': Population,
	'Households': AveOccup,
	'Median Income': MedInc
	}
	return data

st.map(pd.DataFrame(data_train_example[['Latitude', 'Longitude']].values, columns=['lat', 'lon']))

if st.button("Predict"):
	df = pd.DataFrame(get_data(), index = [0])
	st.table(df)
	st.text(f"""
		Prédictions en milliers de dollars US qui devrait se situer entre 0.14999 et 5.00001
		La moyenne étant 2.068558169
		Régression linéaire : {modelLinearRegression_0.predict(df)[0][0]}
		Arbre de décision : {modelDecisionTreeRegressor_0.predict(df)[0]}
		Gradient Boost : {modelGradientBoostingRegressor_0.predict(df)[0]}
		Réseau de neurones : {model_ann_0.predict(scaler.transform(df))[0][0]}
		"""
	)

# Random Forest : {modelRandomForestRegressor_0.predict(df)[0]}