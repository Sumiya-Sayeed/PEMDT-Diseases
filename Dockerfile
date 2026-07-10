# Use the official Jupyter SciPy Notebook image as a base
# This image already includes pandas, numpy, scipy, scikit-learn, and matplotlib
FROM jupyter/scipy-notebook:latest

# Set the working directory in the container
WORKDIR /home/jovyan/work

# Copy the requirements file into the container
COPY requirements.txt /tmp/

# Install any additional dependencies (like jax)
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the entire project repository into the container
# Ensure .dockerignore is used to avoid copying huge/unnecessary files
COPY . /home/jovyan/work/

# Set the correct permissions for the jovyan user
USER root
RUN chown -R jovyan:users /home/jovyan/work
USER jovyan

# Expose the default Jupyter port
EXPOSE 8888

# Command to start the Jupyter server
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
