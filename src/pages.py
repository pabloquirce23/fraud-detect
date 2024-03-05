import nbformat
import re
import streamlit as st
import base64
from io import BytesIO
from components.components import custom_title

def cargar_cuaderno_jupyter(path):
    with open(path, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def limpiar_html(html_content):
    # Eliminar script específico de Google Colab
    cleaned_content = re.sub(r'<script>.*?</script>', '', html_content, flags=re.DOTALL)
    return cleaned_content

def mostrar_cuaderno_jupyter(nb, num_celda_inicio=0, num_celda_final=None):
    for i, cell in enumerate(nb.cells):
        # Si se especificó un número de celda final y se alcanza, detener el bucle
        if num_celda_final is not None and i > num_celda_final:
            break
        # Continuar si el índice de la celda actual aún no ha alcanzado el número de celda de inicio
        if i < num_celda_inicio:
            continue
        
        if cell.cell_type == 'markdown':
            st.markdown(cell.source)
        elif cell.cell_type == 'code':
            st.code(cell.source, language='python')
            for output in cell.outputs:
                if output.output_type == 'execute_result' or output.output_type == 'display_data':
                    if 'text/plain' in output.data:
                        st.text(output.data['text/plain'])
                    if 'image/png' in output.data:
                        base64_img = output.data['image/png']
                        img_bytes = base64.b64decode(base64_img)
                        st.image(img_bytes, use_column_width=True)
                    if 'text/html' in output.data:
                        # Llamar a limpiar_html solo para contenido HTML
                        cleaned_html = limpiar_html(output.data['text/html'])
                        st.markdown(cleaned_html, unsafe_allow_html=True)
                        # st.markdown(output.data['text/html'], unsafe_allow_html=True)
                elif output.output_type == 'stream':
                    st.text(output.text)
                elif output.output_type == 'error':
                    st.error('\n'.join(output.traceback))




def show_analysis_page():
    placeholder = st.empty()
    
    
    # Titulo de la aplicación
    st.markdown(custom_title('Fraud-Detect'), unsafe_allow_html=True)

    st.subheader("Análisis Exploratorio")
    # Aquí puedes añadir más contenido para esta página

    # Cargar el cuaderno Jupyter
    nb_path = 'src/static/notebooks/modelo_tfm.ipynb'
    with st.spinner('Cargando análisis exploratorio y Modelo...'):
        # Cargar y mostrar el contenido pesado aquí
        nb = cargar_cuaderno_jupyter(nb_path)

        # Especificar el inicio y el final
        num_celda_inicio = 1  # Ajusta este valor según sea necesario
        num_celda_final = 33  # Ajusta este valor según sea necesario

        # Mostrar el rango especificado de celdas del cuaderno en Streamlit
        mostrar_cuaderno_jupyter(nb, num_celda_inicio)
    
    placeholder.empty()


def show_model_page():
    placeholder = st.empty()
    

    # st.subheader("Entrenamiento del Modelo")
    # Aquí puedes añadir más contenido para esta página   

    # Cargar el cuaderno Jupyter
    nb_path = 'src/static/notebooks/modelo_tfm.ipynb'
    with st.spinner('Cargando modelo...'):
        # Cargar y mostrar el contenido pesado aquí
        nb = cargar_cuaderno_jupyter(nb_path)

        # Especificar el inicio y el final
        num_celda_inicio = 34  # Ajusta este valor según sea necesario
        # num_celda_final = 25  # Ajusta este valor según sea necesario

        # Mostrar el rango especificado de celdas del cuaderno en Streamlit
        mostrar_cuaderno_jupyter(nb, num_celda_inicio=num_celda_inicio)

    placeholder.empty()