FROM python:3.10.12
# FROM openjdk:11-jdk
RUN apt-get update 
RUN apt-get install -y openjdk-17-jdk
RUN pip install streamlit scikit-learn tensorflow tabula-py tabulate
RUN pip install joblib==1.1.0
COPY model/* /app/model/
COPY src/* /app/
WORKDIR /app
ENTRYPOINT [ "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]