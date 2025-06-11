FROM python:3.8-slim-buster
RUN apt update -y && apt install awscli -y
# Set the working directory
WORKDIR /app

COPY . /app

# Install the required packages
RUN pip install -r requirements.txt
RUN pip install --upgrade accelerate
RUN pip uninstall -y transformers accelerate
RUN pip install transformers accelerate

CMD ["python", "app.py"]