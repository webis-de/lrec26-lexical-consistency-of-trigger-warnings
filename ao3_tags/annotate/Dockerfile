FROM huggingface/transformers-pytorch-gpu:4.41.2

WORKDIR /app

# Get the necessary files to run the job
COPY ./ao3_tags/conf /app/ao3_tags/conf
COPY ./ao3_tags/__init__.py /app/ao3_tags/
COPY ./resources /app/resources
COPY ./ao3_tags/annotate /app/ao3_tags/annotate

RUN pip install --upgrade pip
RUN pip install -r "/app/ao3_tags/annotate/requirements.txt"

# Install packags for quantization
RUN pip install accelerate
RUN pip install -U bitsandbytes

# Add ao3_tags to path
ENV PYTHONPATH="/app/"