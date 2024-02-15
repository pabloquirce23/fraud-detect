import streamlit as st
import pandas as pd
import numpy as np
from tabula.io import read_pdf

st.title("FRAUD DETECTOR")


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
                     st.table(tablas[i])
            else:
               st.write("¡Solo se permiten un máximo de 5 archivos!")
               break



st.markdown("Creadores: Pablo Oller Pérez, Pablo Santos Quirce, Alejandro Castillo Carmona")
