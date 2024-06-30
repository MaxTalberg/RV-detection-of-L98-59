# Start with a base Ubuntu image
FROM ubuntu:20.04

# Set noninteractive installation to avoid getting stuck at prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    wget \
    git \
    bzip2 \
    build-essential  # This might be necessary for compiling things like PolyChord

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh \
    && bash /miniconda.sh -b -p /opt/conda \
    && rm /miniconda.sh

# Add Conda to PATH
ENV PATH="/opt/conda/bin:${PATH}"

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in environment.yml
COPY environment.yml /app/environment.yml
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "l9859-env", "/bin/bash", "-c"]

# Install PolyChord from source
RUN git clone https://github.com/PolyChord/PolyChordLite.git \
    && cd PolyChordLite \
    && python setup.py install \
    && cd .. \
    && rm -rf PolyChordLite

# Define environment variable
ENV NAME l9859-env

# Run main.py when the container launches
CMD ["conda", "run", "--no-capture-output", "-n", "l9859-env", "python", "src/main.py"]
