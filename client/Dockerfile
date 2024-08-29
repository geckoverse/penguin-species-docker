FROM python:3.9-slim

COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY penguins.py /app/penguins.py

CMD ["streamlit", "run", "penguins.py"]