FROM python:3.11-slim-buster

WORKDIR /app

COPY model/requirements.txt ./model/

RUN pip install --upgrade pip

RUN pip install -r ./model/requirements.txt

RUN pip install fastapi fastapi-cli uvicorn

COPY . .

WORKDIR /

EXPOSE 8000

CMD ["fastapi", "run", "app/main.py", "--port", "8000"]