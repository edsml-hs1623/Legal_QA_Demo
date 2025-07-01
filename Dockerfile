FROM python:3.10

WORKDIR /app

ADD . /app
RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["bash", "start.sh"]