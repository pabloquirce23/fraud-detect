import streamlit as st
import pandas as pd
import tabula
import joblib
import tensorflow as tf

# Define las columnas
columnas = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# Carga el modelo de detección de fraude
modelo = joblib.load('model/modelo_fraud_detect.pkl')

# Crea un DataFrame vacío con las columnas definidas
df = pd.DataFrame(columns=columnas)

# Permite al usuario subir archivos PDF
uploaded_files = st.file_uploader("Elige tus archivos PDF", type="pdf", accept_multiple_files=True)

for uploaded_file in uploaded_files:
    if uploaded_file is not None:
        # Lee las tablas del archivo PDF en una lista de DataFrames
        df_temp_list = tabula.read_pdf(uploaded_file, pages='all')
        
        for df_temp in df_temp_list:
            # Comprueba si df_temp es un DataFrame
            if isinstance(df_temp, pd.DataFrame):
                # Asegúrate de que el DataFrame tenga las columnas correctas
                if set(columnas).issubset(df_temp.columns):
                    # Añade el DataFrame al DataFrame principal
                    df = pd.concat([df, df_temp], ignore_index=True)
                else:
                    st.write(f"El archivo {uploaded_file.name} no tiene las columnas correctas.")
            else:
                st.write(f"El archivo {uploaded_file.name} no contiene ninguna tabla.")

# Reemplaza todas las comas (,) por puntos (.) en el DataFrame
df = df.replace(',', '.', regex=True)

# Convierte todas las columnas del DataFrame a tipo float
df = df.astype(float)

# Convierte el DataFrame de pandas a un Tensor de TensorFlow
df_tensor = tf.convert_to_tensor(df.values, dtype=tf.float32)

# Aplica el modelo de detección de fraude al Tensor
if not df.empty:
   df['Class'] = modelo.predict(df_tensor)
   for i in range(len(df)):
       if df['Class'][i] == 0:
           st.write(f"La información de {uploaded_file.name} es probablemente no fraudulenta.")
       else:
           st.write(f"La información de {uploaded_file.name} es probablemente fraudulenta.")
else:
   st.write('No se han subido archivos PDF válidos.')