# our base image
FROM python:3.9-slim

# set working directory inside the image
WORKDIR /app

# copy our requirements
COPY requirements.txt requirements.txt
COPY setup.py setup.py

# install dependencies
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip3 install -e .

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# copy this folder contents to image
COPY . .

# run the application
# CMD [ "python3", "install" , "-e", "."]