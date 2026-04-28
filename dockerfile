# 베이스 이미지 설정
FROM python:3.12.0-slim

RUN apt-get update && apt-get install -y \
    pkg-config \
    libhdf5-dev \
    build-essential \
    ca-certificates \
    curl \
    wget \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

RUN update-ca-certificates

RUN pip install --no-cache-dir certifi


RUN curl -fsSL https://ollama.com/install.sh | sh && \
    if ! command -v ollama > /dev/null; then \
        echo "ollama 설치 실패"; exit 1; \
    else \
        echo "ollama 설치 완료"; \
    fi

# 소스 코드 복사
COPY . /app

# 작업 디렉토리 설정
WORKDIR /app

# 필요 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

RUN echo "export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt" >> $HOME/.bashrc
ENV HF_HUB_DISABLE_SSL_VERIFICATION=1



# 애플리케이션 실행
# CMD ["streamlit", "run", "RAG_Policy_search.py", "--server.port=8501", "--server.address=0.0.0.0"]
CMD ["sh", "-c", "ollama start & while ! nc -z localhost 11434; do sleep 1; done && ollama create EEVE-Korean-10.8B -f ./Modelfile & sleep 5 && streamlit run RAG_Policy_search.py --server.port=8501 --server.address=0.0.0.0"]


