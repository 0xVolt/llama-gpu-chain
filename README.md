# Create a local, GPU-powered inference endpoint for a LLaMa model.

## About this repository

This repository was made to illustrate how a local instance of a llama model could be created and loaded either onto the CPU or GPU. To use the code in this repository, follow the [`Usage`](#usage) section to install the necessary requirements.

## Usage

Run `install.sh`. If you have a GPU on your machine, use the `--gpu` flag to install the correct dependencies for `llama-cpp`. For instance, to install dependencies with GPU support:

1. `chmod +x ./install.sh`
2. `./install.sh`

## References

1. [Running `llama-cpp` inferences on an Nvidia GPU](https://kubito.dev/posts/llama-cpp-linux-nvidia/)