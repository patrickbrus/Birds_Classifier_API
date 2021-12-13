FROM python:3.9

RUN apt-get update -y

WORKDIR /usr/src/app

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY ./ ./

CMD [ "python", "./predict_app.py" ]