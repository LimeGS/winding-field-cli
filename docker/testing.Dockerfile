# Test tooling is a derivative image; production runtime dependencies stay locked.
ARG IMAGE=winding-field:cpu
FROM ${IMAGE}
USER root
RUN uv pip install --python /opt/winding-field/bin/python pytest==8.4.2 && \
    mkdir /src && \
    ln -s /opt/winding-field/lib/python3.12/site-packages/winding_field /src/winding_field
USER 10001:10001
ENTRYPOINT ["python", "-m", "pytest"]
