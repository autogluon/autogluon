FROM python:3.8

RUN python3 -m pip --no-cache-dir install --upgrade wheel setuptools \
    && python3 -m pip --no-cache-dir install --upgrade huggingface-hub awscli

COPY download_hf_models.py ./
COPY download_hf_models.sh ./
RUN chmod +x download_hf_models.sh

CMD ["/bin/bash"]
