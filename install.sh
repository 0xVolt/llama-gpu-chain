# Check if the --gpu flag is provided
if [[ $* == *--gpu* ]]; then
    # If GPU flag is present, install with GPU-related flags
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --verbose
else
    # If GPU flag is not present, install without GPU-related flags
    pip install llama-cpp-python --verbose
fi

# Install other Python dependencies
pip install -r requirements.txt