FROM python:3.10-slim

WORKDIR /app

#RUN apt-get update && apt-get install -y \
#    build-essential \
#    curl \
#    git \
#    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    gcc \
    g++ \
    libglib2.0-0 \
    libgl1 \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY src/ ./src/

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/program/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
