import streamlit as st
import tabula
import pandas as pd
import joblib
import tensorflow as tf
from components.components import custom_title, description


# definición de columnas del dataset
columnas = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']


def show_home_page():
    ### Cabecera  ##################################################################################

    # Imagen Cabecera de la aplicación
    st.image("src/static/image/banner.webp", use_column_width=True, output_format='auto')

    # Titulo de la aplicación
    st.markdown(custom_title('Fraud-Detect'), unsafe_allow_html=True)

    # Breve descripción de la aplicación
    st.markdown(description(), unsafe_allow_html=True)


    st.subheader("Página de Inicio")
    uploaded_files = st.file_uploader("Elige tus archivos PDF", type="pdf", accept_multiple_files=True, key='home_page_file_uploader')

    # Inicializa df solo si hay nuevos archivos para procesar
    if uploaded_files and ('uploaded_files' not in st.session_state or uploaded_files != st.session_state.uploaded_files):
        df = pd.DataFrame(columns=columnas)  # Crear un nuevo DataFrame para los archivos cargados
        
        for uploaded_file in uploaded_files: # Bucle para analizar archivo por archivo en los archivos subidos
            if uploaded_file is not None:
                df_temp_list = tabula.read_pdf(uploaded_file, pages='all') # crea un array de posibles DataFrames con la lectura de las tablas del pdf
                for df_temp in df_temp_list: # Bucle para analizar cada posible dataframe del arra df_temp_list
                    if isinstance(df_temp, pd.DataFrame): # Comprueba si es un DataFrame
                        if set(columnas).issubset(df_temp.columns): # Comprueba si el DataFrame  contiene las mismas columnas que el df donde se van a concatenar
                            df = pd.concat([df, df_temp], ignore_index=True) # Concatena df_temp en df
                        else:
                            st.write(f"El archivo {uploaded_file.name} no tiene las columnas correctas.")
                    else:
                        st.write(f"El archivo {uploaded_file.name} no contiene ninguna tabla.")

        # Configura opción para evitar conversión silenciosa de tipos de datos.
        pd.set_option('future.no_silent_downcasting', True) 
        # Reemplaza comas con puntos en el DataFrame.
        df.replace(',', '.', regex=True, inplace=True) 
        # Convierte todas las columnas a tipo de datos 'float'.
        df = df.astype(float)
        
        columnas_median = [c for c in columnas if c not in ['Time', 'Amount']] # Crea una variable de las columnas sin las columnas 'Time' y 'Amount'
        df['Median'] = df[columnas_median].mean(axis=1) # Crea una nueva columna en df con la mediana de columnas_median
        
        st.session_state['df'] = df  # Almacena el DataFrame procesado en el estado de la sesión
        st.session_state['uploaded_files'] = uploaded_files  # Actualiza la referencia de archivos cargados
