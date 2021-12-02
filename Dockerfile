FROM ubuntu:latest 
RUN apt-get update && apt-get install -y python3.8 python3.8-dev python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN pip3 install -r requirements.txt
EXPOSE 8000
CMD python3.8 /app/api.py