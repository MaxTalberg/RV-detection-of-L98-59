# Use an official Miniconda runtime as a parent image
FROM continuumio/miniconda3

# Set the working directory to /app
WORKDIR /app

# Update the repository sources list and install build tools
RUN apt-get update && apt-get install -y \
    build-essential  # This will install gcc, g++ and make

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in environment.yml
COPY environment.yml /app/environment.yml
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "l9859-env", "/bin/bash", "-c"]

# Install PolyChord from source
RUN git clone https://github.com/PolyChord/PolyChordLite.git
RUN cd PolyChordLite
RUN make
RUN pip install .

# Define environment variable
ENV NAME l9859-env

# Run main.py when the container launches
CMD ["conda", "run", "--no-capture-output", "-n", "l9859-env", "python", "src/main.py"]
