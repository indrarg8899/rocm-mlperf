FROM rocm/rocm-terminal:6.3.0

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir mlperf-loadgen

# Copy source
COPY . .

# Set ROCm environment
ENV HIP_VISIBLE_DEVICES=0
ENV ROCM_PATH=/opt/rocm
ENV PATH=${ROCM_PATH}/bin:${PATH}

ENTRYPOINT ["python", "-m", "src.benchmark"]
CMD ["--config", "configs/resnet50_offline.yml"]
