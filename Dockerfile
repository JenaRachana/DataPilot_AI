FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock /app/

RUN uv sync --no-dev

COPY . /app

EXPOSE 7860

ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["uv", "run", "streamlit", "run", "ui/app.py"]