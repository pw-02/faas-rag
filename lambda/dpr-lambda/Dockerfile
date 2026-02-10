FROM public.ecr.aws/lambda/python:3.12

# Install runtime deps (no torch)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app code + exported ONNX/tokenizer files
COPY app/ ${LAMBDA_TASK_ROOT}/app/

# Lambda handler
CMD ["app.handler.handler"]

#build and test locally with:
#docker build -t dpr-lambda:latest .
