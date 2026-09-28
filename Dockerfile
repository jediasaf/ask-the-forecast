# The MCP server as a network service.
#
# Two stages, so the image that ships holds the installed wheel and nothing that
# built it. The wheel carries the data files, which is the part that matters:
# an image that installs cleanly and then has nothing to read would pass a build
# and fail on the first call.

FROM python:3.12-slim AS build
WORKDIR /src
RUN pip install --no-cache-dir build
COPY pyproject.toml README.md ./
COPY agent agent
COPY mcp_server mcp_server
COPY data data
RUN python -m build --wheel --outdir /dist

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MCP_TRANSPORT=http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000

COPY --from=build /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl \
    && useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app

# Numeric, so Kubernetes can verify runAsNonRoot without resolving a name.
USER 10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"]

ENTRYPOINT ["ask-the-forecast-mcp"]
