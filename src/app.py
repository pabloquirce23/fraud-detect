import streamlit as st
import pandas as pd
import os
import joblib
import tabula

@st.cache(allow_output_mutation=True)
def load_model(model_file):
    model = joblib.load(model_file)
    return model

def apply_models(df, clustering_fraud_detect, modelo_fraud_detect):
   df['Cluster'] = clustering_fraud_detect.predict(df)
   df['Fraud'] = modelo_fraud_detect.predict(df.drop(columns=['Cluster']))
   return df

clustering_fraud_detect = load_model('clustering_fraud_detect.pkl')
modelo_fraud_detect = load_model('modelo_fraud_detect.pkl')

st.sidebar.title('Cargar archivos PDF')
uploaded_files = st.sidebar.file_uploader('Cargar hasta 5 archivos PDF', accept_multiple_files=True)

if uploaded_files:
   for uploaded_file in uploaded_files:
      st.subheader(f'Archivo PDF: {uploaded_file.name}')

      try:
         pdf_df = tabula.read_pdf(uploaded_file)

         result_df = apply_models(pdf_df, clustering_fraud_detect, modelo_fraud_detect)

         st.write(result_df)

      except Exception as e:
         st.error(f'Error al procesar el archivo {uploaded_file.name}: {e}')