# FROM python:3.8.16-alpine3.17

# WORKDIR /app

# ADD req.txt /app

# RUN pip install -r req.txt

# ADD . /app

# CMD [ "python","app.py" ]

# FROM python:3.7-slim AS compile-image

# COPY req.txt .
# RUN python3 -m pip install --user -r req.txt
# RUN python3 -m pip install --user imutils flask-cors torch tritonclient pygame
# FROM python:3.7-slim AS build-image
# RUN apt-get update
# RUN apt-get install -y --no-install-recommends build-essential gcc
# RUN apt-get install ffmpeg libsm6 libxext6  -y
# RUN apt-get install -y python3-opencv
# RUN apt-get install -y libgl1-mesa-dev
# COPY --from=compile-image /root/.local /root/.local
# COPY . .
# # Make sure scripts in .local are usable:
# ENV PATH=/root/.local/bin:$PATH
# CMD ["python","app.py"]

# FROM python:3.8-slim AS compile-image
# RUN apt-get update && apt-get install -y build-essential
# # RUN apt-get install -y cmake
# COPY req.txt .
# # RUN python3 -m pip install --user -r req.txt
# RUN python3 -m pip install --trusted-host files.pythonhosted.org --trusted-host pypi.org --trusted-host pypi.python.org --user numpy==1.19.5
# RUN apt-get install -y cmake
# # RUN pip3 install --user -r req.txt
# FROM python:3.8-slim AS build-image
# #RUN apt-get update && apt-get install -y python3-opencv
# # # RUN apt-get install -y --no-install-recommends build-essential gcc
# # # RUN apt-get install ffmpeg libsm6 libxext6  -y
# # # RUN apt-get install -y python3-opencv
# # # RUN apt-get install -y libgl1-mesa-dev
# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# COPY --from=compile-image /root/.local /root/.local
# COPY . .
# # Make sure scripts in .local are usable:
# ENV PATH=/root/.local/bin:$PATH
# CMD ["python","app.py"]

FROM python:3.8 AS compile-image
RUN apt-get update && apt-get install -y build-essential
# RUN python3 -m pip install --user -r req.txt
# RUN python3 -m pip install --trusted-host files.pythonhosted.org --trusted-host pypi.org --trusted-host pypi.python.org --user -r req.txt
ENV _PYTHON_HOST_PLATFORM linux_armv7l
RUN /usr/local/bin/python3 -m pip install --upgrade pip
RUN python3 -m pip install --trusted-host files.pythonhosted.org --trusted-host pypi.org --trusted-host pypi.python.org --user numpy==1.19.5
RUN apt-get install -y liblapack-dev libblas-dev gfortran
RUN python3 -m pip install --trusted-host files.pythonhosted.org --trusted-host pypi.org --trusted-host pypi.python.org --user scipy==1.8.0
RUN apt-get install -y cmake
# RUN python3 -m pip install --upgrade pip
# RUN python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.0-py3-none-any.whl
# RUN python3 -m pip install --user tensorflow==2.5.0

# COPY reqOriginal.txt .
# RUN python3 -m pip install --user -r reqOriginal.txt

# RUN python3 -m pip install --user scipy==1.8.0
# FROM python:3.10-alpine as base
# RUN apk add --update --virtual .build-deps \ 
# build-base \
# postgresql-dev \
# python3-dev \
# libpq \
# python3-opencv
# COPY req.txt /app/req.txt
# RUN pip install -r /app/req.txt

# FROM python:3.10-alpine
# RUN apk add libpq
# COPY --from=base /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
# COPY --from=base /usr/local/bin/ /usr/local/bin/
# COPY . /app
# ENV PYTHONUNBUFFERED 1

