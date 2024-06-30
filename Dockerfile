# Use an official Miniconda runtime as a parent image
FROM continuumio/miniconda3

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in environment.yml, ignoring errors temporarily
COPY environment.yml /app/environment.yml
RUN conda env create -f environment.yml || echo "Environment creation encountered issues, but continuing..."

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "l9859-env", "/bin/bash", "-c"]

# Install PolyChord from source
RUN git clone https://github.com/PolyChord/PolyChordLite.git \
    && cd PolyChordLite \
    && python setup.py install \
    && cd .. \
    && rm -rf PolyChordLite

# Try installing radvel explicitly, possibly working around earlier issues
RUN conda run -n l9859-env pip install radvel || echo "Failed to install radvel, check compatibility."

# Define environment variable
ENV NAME l9859-env

# Run main.py when the container launches
CMD ["conda", "run", "--no-capture-output", "-n", "l9859-env", "python", "src/main.py"]
