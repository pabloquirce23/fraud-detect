FROM python:3.10.12


# RUN apt-get update 
# RUN apt-get install -y openjdk-17-jdk

# Instala Java
RUN apt-get update 
RUN apt-get install -y openjdk-17-jdk

WORKDIR /app

# Configura JAVA_HOME y añade el directorio bin de Java al PATH
# ENV PATH $JAVA_HOME/bin:$PATH
ENV JAVA_HOME /usr/lib/jvm/java-17-openjdk

# ENV JAVA_HOME /usr/lib/jvm/java-17-openjdk

# Instalar dependencias básicas primero
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install  -r requirements.txt
# RUN pip install --no-cache-dir streamlit

# Luego instalar dependencias de desarrollo
# COPY requirements_dev.txt /app/
# RUN pip install --no-cache-dir -r requirements_dev.txt
# Ahora copiamos el resto de los archivos
COPY src/ /app/
# COPY src/model/ /app/model/
# COPY src/data/* /app/src/data/
# # COPY image/* /app/image/
# COPY src/model/* /app/model/
# COPY src/components/* /app/components/
# COPY src/static/* /app/static/

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
