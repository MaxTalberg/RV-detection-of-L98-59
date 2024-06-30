# Use an official Ubuntu as a parent image
FROM ubuntu:20.04

# Set non-interactive installation mode, to avoid getting prompted
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory to /app
WORKDIR /app

# Install system packages required by Miniconda and PolyChord
# Install system packages required by Miniconda and PolyChord
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    build-essential \ # Includes gcc, g++ and make
    gfortran \
    gcc \ # Explicitly installing gcc
    git \
    && rm -rf /var/lib/apt/lists/*


# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /miniconda \
    && rm miniconda.sh

# Add conda to PATH
ENV PATH=/miniconda/bin:$PATH

# Copy the environment.yml file to the container
COPY environment.yml /app/

# Create the Conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "l9859-env", "/bin/bash", "-c"]

# Copy the rest of your app's source code from the local directory to /app inside the container
COPY . /app

# Install PolyChord from source
RUN git clone https://github.com/PolyChord/PolyChordLite.git \
    && cd PolyChordLite \
    && make \
    && pip install . \
    && cd .. \
    && rm -rf PolyChordLite

# Define environment variable to be used in your application
ENV NAME l9859-env

# Run main.py when the container launches
CMD ["conda", "run", "--no-capture-output", "-n", "l9859-env", "python", "src/main.py"]
