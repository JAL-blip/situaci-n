import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción del costo  ''')
st.image("cosimagen.jpeg", caption="Predicción del costo.")

st.header('Datos personales')

def user_input_features():
  # Entrada
  Presupuesto = st.number_input('Presupuesto:', min_value=0, max_value=5000, value = 0, step = 1)
  Tiempo_invertido = st.number_input('Tiempo invertido:',  min_value=0, max_value=1000, value = 0, step = 1)
  Tipo = st.number_input('Tipo (1,2,3,4,5,6):', min_value=0, max_value=6, value = 0, step = 1)
  Momento = st.number_input('Momento (0(mañana),1(tarde),2(noche)):', min_value=0, max_value=2, value = 0, step = 1)
  No_de_personas = st.number_input('no. personas:', min_value=0, max_value=50, value = 0, step = 1)


  user_input_data = {'Presupuesto': Presupuesto,
                     'Tiempo invertido': Tiempo_invertido,
                     'Tipo': Tipo,
                     'Momento': Momento,
                     'No. de personas': No_de_personas,
                     }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

datos =  pd.read_csv('Costo2.csv.csv', encoding='latin-1')
X = datos.drop(columns='Costo')
y = datos['Costo']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1614372)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*33.5 + b1[1] + b1[2]+ b1[3] + b1[4]

st.subheader('Cálculo del costo')
st.write('El costo es de', prediccion)
