ARG STAMPDESK_DEPS_IMAGE=python:3.11-slim
FROM ${STAMPDESK_DEPS_IMAGE} AS dependencies
USER root
WORKDIR /build/stampdesk
COPY requirements.txt .
RUN python -m pip install --disable-pip-version-check --no-cache-dir --only-binary=:all: -r requirements.txt && python -m pip check

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOST=0.0.0.0 PORT=8000 STAMPDESK_DATA=/var/lib/stampdesk
WORKDIR /opt/stampdesk
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY stampdesk ./stampdesk
COPY static ./static
RUN useradd -U -u 11513 stampdesk && install -d -o stampdesk -g stampdesk /var/lib/stampdesk
USER stampdesk
EXPOSE 8000
CMD ["python", "-m", "stampdesk"]
