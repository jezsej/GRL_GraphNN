FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl unzip htop libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx \
    && apt-get clean

# Copy and install requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set working dir and copy code
WORKDIR /workspace
COPY ./src /workspace/src
COPY ./configs /workspace/configs
COPY ./evaluation /workspace/evaluation
COPY run.sh /workspace/run.sh

RUN chmod +x /workspace/run.sh

CMD ["/workspace/run.sh"]