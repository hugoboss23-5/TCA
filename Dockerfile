FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -e .
EXPOSE 8100
CMD ["python", "tca_mcp/server.py"]
