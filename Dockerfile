# Use an official Ubuntu runtime as a parent image
FROM ubuntu

# Set the working directory to /app
WORKDIR /app

# Install system packages required by Miniconda and PolyChord
# Including gcc, gfortran for PolyChord, and git to clone the repository
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    build-essential \
    gfortran \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc

# Copy the environment.yml file to container
COPY environment.yml /app/environment.yml

# Create the conda environment
RUN /opt/conda/bin/conda env create -f environment.yml

# Activate the environment and configure the shell
SHELL ["conda", "run", "-n", "l9859-env", "/bin/bash", "-c"]

# Install PolyChord from source
RUN git clone https://github.com/PolyChord/PolyChordLite.git \
    && cd PolyChordLite \
    && make \
    && cd .. \
    && pip install .

# Copy the remaining directory contents into the container at /app
COPY . /app

# Define environment variable
ENV NAME l9859-env

# Run main.py when the container launches
CMD ["conda", "run", "--no-capture-output", "-n", "l9859-env", "python", "src/main.py"]
