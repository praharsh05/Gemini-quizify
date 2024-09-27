FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir virtualenv

RUN virtualenv env

RUN /bin/bash -c "source env/bin/activate && pip install --no-cache-dir -r requirements.txt"

COPY . .

ENV GOOGLE_APPLICATION_CREDENTIALS="./service_account_key.json"

EXPOSE 8501

CMD ["/bin/bash", "-c", "source env/bin/activate && streamlit run main.py --server.port=8501 --server.address=0.0.0.0"]