FROM python:3.12.5

COPY main.py /app/
COPY requirements.txt /app/ 
COPY models /app


WORKDIR /app
RUN pip install -r requirements.txt

CMD [ "python", "main.py" ]