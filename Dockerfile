FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY hbv_sirna ./hbv_sirna
RUN pip install --no-cache-dir .

ENTRYPOINT ["hbv-sirna"]
CMD ["--help"]

