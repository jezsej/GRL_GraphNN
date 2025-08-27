FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# ----------------------
# 1. System Dependencies
# ----------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London

RUN apt-get update && \
    apt-get install -y \
    git curl unzip htop build-essential \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx \
    tzdata && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ----------------------
# 2. Set Working Directory
# ----------------------
WORKDIR /app

# ----------------------
# 3. Install PyTorch Geometric dependencies for torch==2.1.0
# ----------------------
RUN pip install \
    torch-scatter==2.1.1 \
    torch-sparse==0.6.17 \
    torch-geometric==2.4.0 \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# ----------------------
# 4. Copy and install other Python requirements
# ----------------------
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ----------------------
# 5. Copy Project Files
# ----------------------
COPY config/ /app/config/
COPY models/ /app/models/
COPY training/ /app/training/
COPY evaluation/ /app/evaluation/
COPY data/ /app/data/
COPY logs/ /app/logs/
COPY utils/ /app/utils/
COPY result/ /app/result/
COPY figures/ /app/figures/
COPY run_loso.py /app/run_loso.py
# COPY run_loso_one_site.py /app/run_loso_one_site.py
COPY run.sh /app/run.sh
# COPY run_parallel.sh /app/run_parallel.sh
# COPY run_loso_one_site.py /app/run_loso_one_site.py
COPY check_imports.py /app/check_imports.py

# ----------------------
# 6. Make Entry Script Executable
# ----------------------
RUN chmod +x /app/run.sh
# RUN chmod +x /app/run_parallel.sh

# ----------------------
# 7. Set PYTHONPATH
# ----------------------
ENV PYTHONPATH="/app:$PYTHONPATH"

# ----------------------
# 8. Entry Command
# ----------------------
CMD ["/app/run.sh"]
# CMD ["/app/run_parallel.sh"]