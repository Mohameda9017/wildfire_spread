# Use the official TensorFlow GPU image as the base
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory inside the container
WORKDIR /workspace

# Create a non-root user 'vscode' with sudo access
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Copy requirements.txt and install packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Note: The user requested to skip additional library installations (numpy, matplotlib, scikit-learn)
# assuming they are already present in the base image or to avoid changes to requirements.txt.
