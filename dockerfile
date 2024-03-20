FROM python:3.10

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get upgrade -y
RUN apt-get install build-essential git python3 python3-pip wget --no-install-recommends -y
RUN apt-get install ffmpeg libsm6 libxext6 libxrender1 libglib2.0-0 --no-install-recommends -y
RUN apt-get install libicu-dev libcairo2-dev libtesseract-dev tesseract-ocr --no-install-recommends -y
RUN apt-get install -y build-essential git python3 python3-pip wget libsm6 libxext6 libxrender1 libglib2.0-0
RUN apt install libicu-dev libicu-dev libcairo2-dev libtesseract-dev tesseract-ocr -y

RUN make build

EXPOSE 8000
ENV NAME env_file

WORKDIR /app/src
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]