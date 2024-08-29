FROM python:3.9-slim

COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY penguins.py penguins.py
COPY model.pkl /app/model.pkl

CMD ["uvicorn", "--host", "0.0.0.0", "penguins:app"]