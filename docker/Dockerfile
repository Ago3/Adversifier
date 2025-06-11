FROM python:3.8-slim

# Set environment variables to reduce interactive prompts and cache issues
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install pip packages
RUN pip install --upgrade pip && \
    pip install \
        nltk \
        transformers==4.3.0 \
        scikit-learn==0.24.1 \
        packaging==21.3 \
        'torchmetrics<0.8'
