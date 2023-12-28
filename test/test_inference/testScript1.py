def checkGPU(tensorflow):
    if tensorflow == True:
        import tensorflow as tf
        print("Number of GPUs available with tensorflow:", len(tf.config.list_physical_devices('GPU')))
    else:
        import torch
        print('Checking if the GPU is available with PyTorch:', torch.cuda.is_available())