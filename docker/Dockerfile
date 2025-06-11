# Set up environment name
ENV ENV_NAME=aaa
ENV PYTHON_VERSION=3.8

# Create Conda environment and activate it
RUN conda create -n $ENV_NAME python=$PYTHON_VERSION -y \
 && conda init bash

# Use bash shell to activate Conda env for all future RUN commands
SHELL ["conda", "run", "-n", "aaa", "/bin/bash", "-c"]

# Install pip packages inside the conda environment
RUN pip install \
    nltk \
    transformers==4.3.0 \
    scikit-learn==0.24.1 \
    packaging==21.3

# Set default command to open bash in the conda environment
CMD ["conda", "run", "-n", "aaa", "bash"]