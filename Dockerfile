# Use an official Miniconda runtime as a parent image
FROM continuumio/miniconda3

# Set the working directory to /app
WORKDIR /app

# Update the repository sources list and install build tools
# It's a good practice to clean up the apt cache by removing /var/lib/apt/lists
# This reduces the image size since the cache is not stored in the layer
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    gcc \
 && rm -rf /var/lib/apt/lists/*

# Copy only the environment.yml initially to avoid cache invalidation by other file changes
COPY environment.yml /app/
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "l9859-env", "/bin/bash", "-c"]

# Copy the current directory contents into the container at /app
# Doing this after environment setup utilizes Docker cache layers more efficiently
COPY . /app

# Install PolyChord from source
RUN git clone https://github.com/PolyChord/PolyChordLite.git

# Define environment variable
ENV NAME l9859-env

# Run main.py when the container launches
CMD ["conda", "run", "--no-capture-output", "-n", "l9859-env", "python", "src/main.py"]
