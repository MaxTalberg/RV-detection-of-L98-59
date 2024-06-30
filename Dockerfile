# Use an official Miniconda runtime as a parent image
FROM continuumio/miniconda3:4.10.3

# Set the working directory to /app
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    git

COPY . /app

RUN conda env create -f /app/environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "l9859-env", "/bin/bash", "-c"]

# Ensure Python output is set straight to the terminal without buffering
ENV PYTHONUNBUFFERED=1
# Copy the current directory contents into the container at /app

# Define environment variable
ENV NAME l9859-env

# Run main.py when the container launches
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "l9859-env", "python", "/app/src/main.py"]
