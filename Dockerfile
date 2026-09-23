# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-compile ".[studio]"


FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --gid 10001 arf \
    && useradd --uid 10001 --gid arf --no-create-home \
        --home-dir /app --shell /usr/sbin/nologin arf \
    && mkdir -p /app/data \
    && chown arf:arf /app/data

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

USER arf

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request; assert urllib.request.urlopen('http://127.0.0.1:7860/', timeout=2).status == 200"]

CMD ["arf", "studio", "--data-dir", "/app/data", "--host", "0.0.0.0", "--port", "7860", "--no-open", "--allow-network"]
