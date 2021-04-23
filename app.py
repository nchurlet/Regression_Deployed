import streamlit as st
import pickle
import pandas as pd
from PIL import Image

# modèles:
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import sklearn.preprocessing
from tensorflow.keras.models import Sequential
#from keras.models import load_model
import tensorflow as tf


# Sidebar - this sidebar allows the user to set the parameters that will be used by the model to create the prediction.
st.sidebar.header('Renseignez les caractéristiques des biens du quartier pour obtenir une évaluation des prix des biens')

# @st.cache
def load_decision_tree():
	global modelDecisionTreeRegressor_0
	modelDecisionTreeRegressor_0 = pickle.load( open( "modelDecisionTreeRegressor_0.p", "rb" ))
# @st.cache
def load_gradient_boost():
	global modelGradientBoostingRegressor_0
	modelGradientBoostingRegressor_0 = pickle.load( open( "modelGradientBoostingRegressor_0.p", "rb" ))

# @st.cache
def load_linear_regressor():
	global modelLinearRegression_0
	modelLinearRegression_0 = pickle.load( open( "modelLinearRegression0.p", "rb" ))

# @st.cache
def load_scaler():
	global scaler 
	scaler = pickle.load( open("scaler.p", "rb" ))

# @st.cache
def load_neurone_network():
	global model_ann_0
	model_ann_0 = tf.keras.models.load_model('./modelann.h5')

# @st.cache
def  load_data():
	global data_train_example
	data_train_example = pickle.load( open( "./data_train_example.p", "rb" ))

# @st.cache
def load_random_forest():
	global modelRandomForestRegressor_0
	modelRandomForestRegressor_0 = pickle.load( open( "modelRandomForestRegressor0.p", "rb" ))

# @st.cache
def load_objects():

	load_decision_tree()
	load_gradient_boost()
	load_linear_regressor()
	load_scaler()
	load_neurone_network()
	# load_random_forest()
	load_data()

load_objects()

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