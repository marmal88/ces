FROM python:3.8-slim-buster

WORKDIR /app

COPY docker/requirements.txt .

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["sh", "-c", "python -m streamlit run src/app.py --server.port=6006 --server.address=0.0.0.0"]