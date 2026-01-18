FROM ghcr.io/developmentseed/titiler:0.24.2

# The base image already includes titiler and rasterio/GDAL stack.
RUN pip install fastapi==0.115.0 uvicorn[standard]==0.30.6 \
    && pip install mmcv==2.1.0 mmengine==0.10.4 mmsegmentation==1.2.2 \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY app.py /app/app.py
RUN mkdir -p /app/static

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
