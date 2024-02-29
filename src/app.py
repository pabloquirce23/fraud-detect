import streamlit as st
import pandas as pd
import tabula
import joblib
import tensorflow as tf

# definición de columnas del dataset
columnas = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# carga del modelo de detección de fraude
modelo = joblib.load('model/modelo_fraud_detect.pkl')

# carga del modelo de clustering
modelo_clustering = joblib.load('model/clustering_fraud_detect.pkl')

# creación del dataframe
df = pd.DataFrame(columns=columnas)

# boton de subida de archivos (no hay límite y se pueden eliminar en la propia página)
uploaded_files = st.file_uploader("Elige tus archivos PDF", type="pdf", accept_multiple_files=True)

for uploaded_file in uploaded_files:
    if uploaded_file is not None:
        # lectura de las tablas de los archivos
        df_temp_list = tabula.read_pdf(uploaded_file, pages='all')
        
        for df_temp in df_temp_list:
            # comprueba si es un dataframe
            if isinstance(df_temp, pd.DataFrame):
                # comprueba si tiene las columnas correctas
                if set(columnas).issubset(df_temp.columns):
                    # concatenación al dataframe principal
                    df = pd.concat([df, df_temp], ignore_index=True)
                else:
                    st.write(f"El archivo {uploaded_file.name} no tiene las columnas correctas.")
            else:
                st.write(f"El archivo {uploaded_file.name} no contiene ninguna tabla.")

# reemplaza las comas por puntos para poder convertir en float
df = df.replace(',', '.', regex=True)

# conversión de los datos de los archivos a float
df = df.astype(float)

# cálculo de la media para poder aplicar el modelo de clustering
columnas_median = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
                   'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 
                   'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']
df['Median'] = df[columnas_median].mean(axis=1)

# columnas necesarias para la aplicación del modelo de clustering
df_clustering = df[['Median', 'Amount']]

# conversión de dataframe a tensor
df_tensor = tf.convert_to_tensor(df[columnas].values, dtype=tf.float32)  # Solo selecciona las columnas originales aquí
df_clustering_tensor = tf.convert_to_tensor(df_clustering.values, dtype=tf.float32)

# aplicación de los modelos
if not df.empty:
   df['Class'] = modelo.predict(df_tensor)
   df['Cluster'] = modelo_clustering.predict(df_clustering_tensor)
   
   for i in range(len(df)):
       if df['Class'][i] == 0:
           st.write(f"La información de {uploaded_file.name} es probablemente no fraudulenta, pertenece al cluster {df['Cluster'][i]}, y su mediana es {df['Median'][i]}.")
       else:
           st.write(f"La información de {uploaded_file.name} es probablemente fraudulenta, pertenece al cluster {df['Cluster'][i]}, y su mediana es {df['Median'][i]}.")
else:
   st.write('No se han subido archivos PDF válidos.')