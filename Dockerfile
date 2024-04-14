# 
FROM python:3.10

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./ /code/app

# 
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
WORKDIR /code/app
EXPOSE 5500
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
CMD ["python", "main.py"]