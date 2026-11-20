ARG CELLTRACE_DEPS_IMAGE=python:3.11-slim
FROM ${CELLTRACE_DEPS_IMAGE} AS dependencies
USER root
WORKDIR /build/celltrace
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements.txt && python -m pip check

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOST=0.0.0.0 PORT=8000 CELLTRACE_DATA=/var/lib/celltrace
WORKDIR /opt/celltrace
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY celltrace ./celltrace
COPY static ./static
COPY app.py .
RUN useradd -U -u 11490 celltrace && install -d -o celltrace -g celltrace /var/lib/celltrace
USER celltrace
EXPOSE 8000
CMD ["python", "app.py"]
