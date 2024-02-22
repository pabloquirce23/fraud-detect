import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tabula.io import read_pdf


modelo = joblib.load("model/modelo_fraud_detect.pkl")
tablas_fraude = []

st.markdown("""
   <style>

   .title {
      text-align: center;
      color: #FF4B4B;
   }
   .footer{
      border: solid 1px #FF4B4B; 
      text-align: center;
      height: 100%;
      background-color: #FF4B4B;
      font-weight: bold;      
   }
   </style>
""", unsafe_allow_html=True)
st.markdown("<h1 class='title'>FRAUD DETECTOR</h1>", unsafe_allow_html=True)


st.write("Vivimos en una sociedad en la que el fraude es el pan de cada día, desde engañar a personas mayores abusando de su desconocimiento de las nuevas tecnologías a intentar saltarse la ley de todas las maneras posibles para conseguir el mayor rédito posible. Con nuestra aplicación queremos ayudar a encontrar a los infractores de actividades tan inmorales y así poder llevarlos frente a la ley.")

files = []
tablas = []

uploaded_files = st.file_uploader("Inserta el archivo AQUÍ", accept_multiple_files=True)

if uploaded_files is not None:
   if len(files) + len(uploaded_files) > 5:
      st.write("¡Solo se permiten un máximo de 5 archivos!")
   else:
      for uploaded_file in uploaded_files:
            if uploaded_file != None:
               if len(files) < 5:
                  files.append(uploaded_file)
                  tablas = read_pdf(uploaded_file, pages = "all")
                  tablas.append(tablas)
                  st.write(f"Archivo: {uploaded_file.name}")
                  for i in range(len(tablas)-1):
                     st.write(f"Tabla {i+1}")
                     # st.table(tablas[i])
            else:
               st.write("¡Solo se permiten un máximo de 5 archivos!")
               break




for tabla in tablas:
   df = pd.DataFrame(tabla)

   st.table(df)
   fraudes = []   
   # tablaCSV = df.to_csv("tablaCSV", index=False)
  
   for fila in df:
      fraudes.append(modelo.predict(fila))
   df['fraude'] = fraudes
  
   # Filtrar las filas donde 'fraude' es igual a 1
   df_fraude = df[df['fraude'] == 1]

   tablas_fraude.append(df_fraude)
   st.table(df_fraude)


st.markdown('<div class="footer">Creadores: Pablo Oller Pérez, Pablo Santos Quirce, Alejandro Castillo Carmona</div>', unsafe_allow_html=True)
