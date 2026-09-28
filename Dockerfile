FROM public.ecr.aws/lambda/python:3.12

COPY requirements-inference.txt ${LAMBDA_TASK_ROOT}/requirements-inference.txt
COPY deployment/certs/ /tmp/bb8-certs/
RUN if compgen -G "/tmp/bb8-certs/*.crt" > /dev/null; then \
        cp /tmp/bb8-certs/*.crt /etc/pki/ca-trust/source/anchors/ && \
        update-ca-trust; \
    fi && \
    python -m pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r ${LAMBDA_TASK_ROOT}/requirements-inference.txt && \
    rm -rf /tmp/bb8-certs

COPY api ${LAMBDA_TASK_ROOT}/api
COPY inference ${LAMBDA_TASK_ROOT}/inference
COPY models ${LAMBDA_TASK_ROOT}/models
COPY tokenizer ${LAMBDA_TASK_ROOT}/tokenizer
COPY deployment/model ${LAMBDA_TASK_ROOT}/deployment/model

ENV BB8_MODEL_DIR=${LAMBDA_TASK_ROOT}/deployment/model \
    BB8_TORCH_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

CMD ["api.handler.lambda_handler"]
