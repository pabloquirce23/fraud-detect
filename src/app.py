import streamlit as st
import pandas as pd
import tabula
import joblib
import tensorflow as tf
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



# Estilo CSS para el título y pie de página
st.write("""
    <style>
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
        
    }
    </style>
""", unsafe_allow_html=True)

st.image("img/banner.webp", use_column_width=True, output_format='auto')
st.markdown("""
   <style>
   .title {
      text-align: center;
      color: #1a4ed8;
   }
   
   </style>
""", unsafe_allow_html=True)



# inicio de fraud detect
st.markdown("<h1 class='title'>Fraud-Detect</h1>", unsafe_allow_html=True)



# Breve descripción de la aplicación
st.write("**Fraud Detect** es una aplicación web diseñada para abordar de manera eficiente y precisa la detección de posibles fraudes bancarios. Su funcionalidad radica en la capacidad de procesar documentos en formato PDF, extrayendo las tablas contenidas en ellos mediante su lector integrado. A partir de los datos recopilados en estas tablas, la aplicación lleva a cabo un exhaustivo análisis para identificar posibles irregularidades financieras que puedan indicar la presencia de actividades fraudulentas entre una lista de clientes. Además muestra una sucesión de gráficas con datos que pueden ser de utilidad para el usuario.")

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

st.divider()

# aplicación de los modelos
if not df.empty:
   # Switcher para mostrar o no las gráficas
   on = st.toggle('Mostrar las gráficas')
   df['Class'] = modelo.predict(df_tensor)
   df['Cluster'] = modelo_clustering.predict(df_clustering_tensor)

   # creación del dataframe para guardar los resultados
   results_df = pd.DataFrame(columns=["Nombre PDF", "Fila PDF", "Detección Fraude", "Cluster"])

   # diccionario para definir las etiquetas de la columna cluster
   cluster_labels = {0: "PG", 1: "LS", 2: "BN", 3: "LN", 4: "IN"}

   # bucle para añadir los resultados al dataframe
   for i in range(len(df)):
       new_row = pd.DataFrame({"Nombre PDF": [uploaded_file.name], 
                               "Fila PDF": [i], 
                               "Detección Fraude": ["NO FRAUDE ✅" if df['Class'][i] == 0 else "FRAUDE ❌"], 
                               "Cluster": [cluster_labels[df['Cluster'][i]]]})
       results_df = pd.concat([results_df, new_row], ignore_index=True)

   st.dataframe(results_df)

   st.divider()

   # gráfica que relaciona la predicción con los clusters (sorted para que salgan de menor a mayor)
   clusters = sorted(df['Cluster'].unique())

   # diccionario para mapear los números del cluster a las etiquetas preferidas
   cluster_labels_2 = {0: "Personal Growth", 1: "Leisure", 2: "Basic Necessities", 3: "Loans", 4: "Investments"}

   fig, axs = plt.subplots(1, len(clusters), figsize=(10, 15))

   # definición de los colores para "Fraud" y "Not Fraud"
   colors = ['darkblue', 'lightblue']

   # se crea una gráfica circular por cada cluster
   for i, cluster in enumerate(clusters):
       # filtra el dataframe al cluster pertinente
       df_cluster = df[df['Cluster'] == cluster]

       # conteo de casos de fraude
       fraud_counts = df_cluster['Class'].value_counts()

       # cálculo del porcentaje de fraud y not fraud
       not_fraud_percentage = fraud_counts.get(0, 0) / fraud_counts.sum() * 100
       fraud_percentage = 100 - not_fraud_percentage

       # creación de las etiquetas
       labels = [f'Fraud {fraud_percentage:.1f}%', f'Not Fraud {not_fraud_percentage:.1f}%']

       # creación de la gráfica
       axs[i].pie([fraud_percentage, not_fraud_percentage], labels=labels, startangle=90, colors = colors)

       # alternación de la posición de título y leyenda
       if i % 2 == 0:
           axs[i].set_title(cluster_labels_2[cluster], y=1.1)
           axs[i].legend(labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
       else:
           axs[i].set_title(cluster_labels_2[cluster], y=-0.1)
           axs[i].legend(labels, loc='lower center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True, ncol=5)
   if on:
      st.pyplot(fig)
      st.divider()


   # Crea un gráfico de dispersión para visualizar los clusters
   plt.figure(figsize=(10, 6))

   for cluster in df['Cluster'].unique():

       # Filtra los datos por cluster
       cluster_data = df[df['Cluster'] == cluster]

       # Plotea los datos con un color diferente para cada cluster
       plt.scatter(cluster_data['Median'], cluster_data['Amount'], label=f'Cluster {cluster}')

   plt.title('Distribución de Transacciones por Clusters')
   plt.xlabel('Mediana de V1-V28')
   plt.ylabel('Amount')
   plt.legend()
   if on:
      st.pyplot(plt)
      st.divider()
else:
   st.write('No se han subido archivos PDF válidos.')


with st.sidebar:
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

# Pie de página con información de los creadores
st.markdown('Creadores:')
st.page_link("https://www.linkedin.com/in/pablo-oller-perez-7995721b2", label="**Pablo Oller Pérez**")
st.page_link("https://www.linkedin.com/in/pablo-oller-perez-7995721b2", label="**Pablo Santos Quirce**")
st.page_link("https://www.linkedin.com/in/pablo-oller-perez-7995721b2", label="**Alejandro Castillo Carmona**")
