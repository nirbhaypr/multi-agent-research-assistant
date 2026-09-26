FROM ghcr.io/astral-sh/uv:0.12.5 AS uv
FROM python:3.14-slim
COPY --from=uv /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
RUN useradd --create-home researcher
USER researcher
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["multi-agent-research-assistant", "serve", "--host", "0.0.0.0"]

