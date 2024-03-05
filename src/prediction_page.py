import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from components.components import custom_title

columnas = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# carga del modelo de detección de fraude
modelo = joblib.load('src/model/modelo_fraud_detect.pkl')

# carga del modelo de clustering
modelo_clustering = joblib.load('src/model/clustering_fraud_detect.pkl')

def show_prediction_page():
    # Titulo de la aplicación
    st.markdown(custom_title('Fraud-Detect'), unsafe_allow_html=True)

    st.subheader("Predicciones de Fraude")

    # Verifica que el DataFrame exista y no esté vacío
    if 'df' in st.session_state and not st.session_state['df'].empty:
        df = st.session_state['df']  # Acceso directo al DataFrame
        # Procede con la lógica de predicción usando 'df'
    
        # columnas necesarias para la aplicación del modelo de clustering
        df_clustering = df[['Median', 'Amount']]

        # conversión de dataframe a tensor
        df_tensor = tf.convert_to_tensor(df[columnas].values, dtype=tf.float32)  # Solo selecciona las columnas originales aquí

        df_clustering_tensor = tf.convert_to_tensor(df_clustering.values, dtype=tf.float32)

        st.divider()

        # aplicación de los modelos
        if not df.empty:
            # Switcher para mostrar o no las gráficas
            # on = st.toggle('Mostrar las gráficas')
            df['Class'] = modelo.predict(df_tensor)
            df['Cluster'] = modelo_clustering.predict(df_clustering_tensor)

            # creación del dataframe para guardar los resultados
            results_df = pd.DataFrame(columns=["Detección Fraude", "Cluster"])

            # diccionario para definir las etiquetas de la columna cluster
            cluster_labels = {0: "PG", 1: "LS", 2: "BN", 3: "LN", 4: "IN"}

            # bucle para añadir los resultados al dataframe
            for i in range(len(df)):
                new_row = pd.DataFrame({"Detección Fraude": ["NO FRAUDE ✅" if df['Class'][i] == 0 else "FRAUDE ❌"], 
                                        "Cluster": [cluster_labels[df['Cluster'][i]]]})
                results_df = pd.concat([results_df, new_row], ignore_index=True)
            
            # Centrado de tabla, no relativo
            col1, col2 = st.columns([4, 6]) 
            with col2:
                st.dataframe(results_df)

            st.divider()


            # gráfica que relaciona la predicción con los clusters (sorted para que salgan de menor a mayor)
            clusters = sorted(df['Cluster'].unique())

            # diccionario para mapear los números del cluster a las etiquetas preferidas
            cluster_labels_2 = {0: "PG", 1: "LS", 2: "BN", 3: "LN", 4: "IN"}

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
                axs[i].pie([fraud_percentage, not_fraud_percentage], startangle=90, colors = colors)

                # alternación de la posición de título y leyenda
                if i % 2 == 0:
                    axs[i].set_title(cluster_labels_2[cluster], y=1.1)
                    axs[i].legend(labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
                else:
                    axs[i].set_title(cluster_labels_2[cluster], y=-0.1)
                    axs[i].legend(labels, loc='lower center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True, ncol=5)
            # if on:
            st.pyplot(fig)
            st.divider()


            # Crea un gráfico de dispersión para visualizar los clusters
            plt.figure(figsize=(10, 6))

            for cluster in df['Cluster'].unique():

                # Filtra los datos por cluster
                cluster_data = df[df['Cluster'] == cluster]

                # Plotea los datos con un color diferente para cada cluster
                plt.scatter(cluster_data['Median'], cluster_data['Amount'], label=f'{cluster_labels_2[cluster]}')

            plt.title('Distribución de Transacciones por Clusters')
            plt.xlabel('Mediana de V1-V28')
            plt.ylabel('Amount')
            plt.legend()
            # if on:
            st.pyplot(plt)
            st.divider()

    else:
        st.error('No se ha cargado ningún DataFrame.')
