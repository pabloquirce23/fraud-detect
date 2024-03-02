import streamlit as st
import pandas as pd
import tabula
import joblib
import tensorflow as tf
import openai
import matplotlib.pyplot as plt
import seaborn as sns

# inicio de fraud detect
st.markdown("<h1 class='title'>FraudDetect</h1>", unsafe_allow_html=True)

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
                # shn kgm
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
   
   # testeo de gráficas con matplotlib
   fig, axs = plt.subplots(3, figsize=(10, 15))

   # gráfica que relaciona la predicción con los clusters
   sns.countplot(x='Cluster', hue='Class', data=df, ax=axs[0])
   axs[0].set_title('Relación entre el tipo de cluster y si es fraude')

   # gráfica que relaciona la predicción con el amount
   sns.countplot(x='Class', y='Amount', data=df, ax=axs[1])
   axs[1].set_title('Relación entre si es fraude y la columna Amount')

   # gráfica que relaciona los clusters con el amount
   sns.countplot(x='Cluster', y='Amount', data=df, ax=axs[2])
   axs[2].set_title('Relación entre el tipo de cluster y la columna Amount')

   st.pyplot(fig)

   # muestra de mensaje placeholder
   for i in range(len(df)):
       if df['Class'][i] == 0:
           st.write(f"La información de {uploaded_file.name} es probablemente no fraudulenta, pertenece al cluster {df['Cluster'][i]}, y su mediana es {df['Median'][i]}.")
       else:
           st.write(f"La información de {uploaded_file.name} es probablemente fraudulenta, pertenece al cluster {df['Cluster'][i]}, y su mediana es {df['Median'][i]}.")
else:
   st.write('No se han subido archivos PDF válidos.')

# inicio del chatbot
st.markdown("<h1 class='title'>Eto'o Bot</h1>", unsafe_allow_html=True)

# api key de openai
openai.api_key = st.secrets["OPENAI_API_KEY"]

# selección del modelo con el que queremos trabajar
if "openai_model" not in st.session_state:
    # 00475-AEDF-52510-2
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# inicialización del histórico del chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# mostrar los mensajes del histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# reacción al input del usuario
if prompt := st.chat_input("¿Cómo va el asunto?"):
    # muestra el mensaje del usuario en su contenetendor de mensaje
    with st.chat_message("user"):
        st.markdown(prompt)
    # añade el mensaje del usuario al histórico
    st. session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        # placeholder vacío
        message_placeholder = st.empty()
        full_response = ""
        # llamada de la api de openai
        for response in openai.ChatCompletion.create(
            # se pasa el modelo y el histórico
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            # este parámetro hace que vaya escribiendo poco a poco la respuesta
            stream=True,
        ):
            # se añade una parte de la respuesta del chatbot en cada iteración del bucle
            full_response += response.choices[0].delta.get("content", "")
            # enseña lo que hay de respuesta
            message_placeholder.markdown(full_response + "| ")
        message_placeholder.markdown(full_response)
    # añadido de la respuesta al histórico
    st.session_state.messages.append({"role": "assistant", "content": full_response})