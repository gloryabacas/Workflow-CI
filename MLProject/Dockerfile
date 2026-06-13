FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install mlflow==2.13.0 scikit-learn==1.4.2 pandas==2.2.2 joblib==1.4.2 matplotlib==3.8.4 dagshub
CMD ["python", "modelling.py"]
