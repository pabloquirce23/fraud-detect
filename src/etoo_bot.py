import streamlit as st
import openai
from components.components import custom_title


def show_etoobot_page():
        # Titulo de la aplicación
        st.markdown(custom_title('Fraud-Detect'), unsafe_allow_html=True)

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
        if prompt := st.chat_input("Escriba aquí su consulta"):
            # muestra el mensaje del usuario en su contenetendor de mensaje
            with st.chat_message("user"):
                st.markdown(prompt)
            # añade el mensaje del usuario al histórico
            st.session_state.messages.append({"role": "user", "content": prompt})

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