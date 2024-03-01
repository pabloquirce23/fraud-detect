FROM python:3.10.12
# FROM openjdk:11-jdk
RUN apt-get update 
RUN apt-get install -y openjdk-17-jdk
RUN pip install streamlit scikit-learn tensorflow tabulate
RUN pip install joblib==1.3.2
RUN pip install openai==0.28
RUN pip install tabula-py==2.2.0
ENV JAVA_HOME /usr/lib/jvm/java-17-openjdk
COPY model/* /app/model/
COPY src/* /app/
WORKDIR /app
ENTRYPOINT [ "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]