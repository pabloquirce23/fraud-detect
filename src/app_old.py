import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import shap
import joblib
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data/creditcard.csv")
df = df.sample(frac=0.01, random_state=42)

# Estilos  ###################################################################################

style_width = """
    <style>
        .appview-container  .main  .block-container{
            max-width: 60%;
        }
    </style>
    """
st.markdown(style_width, unsafe_allow_html=True)



# Cabecera  ##################################################################################
st.title('Fraud Credit Card')

st.markdown("""<br>""", unsafe_allow_html=True)
st.write('Todos los datos sobre los empleados en una aplicación')
# st.write(df)

# Seleccionar las columnas V1 a V28 y mostrar las primeras 5 filas
v_features = df.filter(regex='^V[0-9]+').head(60)

# Mostrar las características seleccionadas
st.dataframe(v_features, height=700)

if not df.empty:
    st.divider()
    with st.container():
        class_counts = df['Class'].value_counts()
        # Estableciendo el estilo de Seaborn
        sns.set_style("whitegrid")

        # Creando el gráfico de barras con Seaborn
        # plt.figure(figsize=(8, 5))
        # sns.barplot(x=class_counts.index, y=class_counts.values, palette="Blues_d")

        # plt.title('Distribución de Clases (0: Legítimo, 1: Fraude)', fontsize=16)
        # plt.xlabel('Clase', fontsize=14)
        # plt.ylabel('Número de Instancias', fontsize=14)
        # plt.xticks(ticks=[0, 1], labels=['Legítimo', 'Fraude'], fontsize=12)
        # ticks = [1, 10, 100, 1000, 10000, 100000]
        # plt.yticks(ticks, labels=[str(tick) for tick in ticks])
        # # plt.yticks(fontsize=12)
        # plt.yscale('log')
        # st.pyplot(plt.gcf())
        # plt.close()
        # Crear el gráfico de barras
        fig = px.bar(x=class_counts.index, y=class_counts.values, log_y=True, labels={'x': 'Clase', 'y': 'Número de Instancias'},
                    title='Distribución de Clases (0: Legítimo, 1: Fraude) con Escala Logarítmica')

        # Mejorar la presentación
        fig.update_layout(
            xaxis_tickmode='array',
            xaxis_tickvals=[0, 1],
            xaxis_ticktext=['Legítimo', 'Fraude'],
            # Ajustes de tamaño
            height=600,
            title_x=0.25,
            title_font=dict(size=22)
        )

        # Añadir anotación para explicar la escala logarítmica
        fig.add_annotation(
            x=0.5, y=-0.2,
            xref="paper", yref="paper",
            text="Escala logarítmica: cada paso aumenta diez veces",
            showarrow=False,
            font=dict(size=12),
            align="center"
        )
        
        # Para asegurar que la anotación sea visible, ajusta el margen inferior
        fig.update_layout(margin=dict(b=120))  # Ajusta el valor según sea necesario
        
        # Mostrar el gráfico
        st.plotly_chart(fig, use_container_width=True)


    st.divider()
    with st.container():
        # Suponiendo que 'data' es tu DataFrame y ya lo has cargado con pd.read_csv
        # Asegúrate de incluir todas las columnas relevantes, aquí asumimos que 'Class' está incluida

        # Calculando la matriz de correlación
        correlation_matrix = df.astype('float32').corr()

        # Creando el mapa de calor
        plt.figure(figsize=(18, 16))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
        plt.title('Matriz de Correlación de Características')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        st.pyplot(plt.gcf())
        plt.close()


    st.divider()
    with st.container():
        # Crear una gráfica de caja para cada característica
        plt.figure(figsize=(16, 10))
        sns.boxplot(data=df, orient='h', palette='Set2')
        plt.title('Distribución de Características')

        # Seleccionar un subconjunto de características de interés
        features_of_interest = ['Time', 'Amount'] + ['V%d' % i for i in range(1, 5)]  # Ejemplo: 'Time', 'Amount', 'V1', 'V2', 'V3', 'V4'

        # Filtrar el DataFrame para incluir solo esas características
        df_subset = df[features_of_interest]

        # Ahora, crear la gráfica de caja para este subconjunto de datos
        plt.figure(figsize=(16, 10))
        sns.boxplot(data=df_subset, orient='h', palette='Set2')
        plt.title('Distribución de Características Seleccionadas')


        # Seleccionar características numéricas para estandarizar
        # features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        # Estandarizar las características
        # scaler = StandardScaler()
        # df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

        # Crear boxplots para un subconjunto de características estandarizadas
        # plt.figure(figsize=(16, 10))
        # sns.boxplot(data=df_scaled[['Time', 'Amount']] + df_scaled.filter(regex='^V[0-9]+').columns.tolist(), orient='h', palette='Set2')
        # plt.title('Distribución de Características Estandarizadas')
        
        
        st.pyplot(plt.gcf())
        plt.close()