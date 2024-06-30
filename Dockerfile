# Use an official Miniconda runtime as a parent image
FROM continuumio/miniconda3

# Set the working directory to /app
WORKDIR /app

# Copy the environment.yml file into the container at /app
COPY environment.yml /app/environment.yml

# Update the repository sources list, install build tools, and adjust permissions
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
 && rm -rf /var/lib/apt/lists/* \
 && chmod -R 777 /app  # Adjust permissions

# Create the Conda environment
RUN conda env create -f environment.yml

# Set the SHELL to use the new environment
SHELL ["conda", "run", "-n", "l9859-env", "/bin/bash", "-c"]

# Copy the rest of your application into the container at /app
COPY . /app

# Install PolyChord from source if necessary
# RUN git clone https://github.com/PolyChord/PolyChordLite.git \
#     && cd PolyChordLite \
#     && make \
#     && pip install . \
#     && cd .. \
#     && rm -rf PolyChordLite

# Define environment variable
ENV NAME l9859-env

# Run main.py when the container launches
CMD ["python", "src/main.py"]
