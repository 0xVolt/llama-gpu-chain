# %% [markdown]
# <a href="https://colab.research.google.com/github/0xVolt/llama-gpu-chain/blob/main/llama_cpp_gpu_inference.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # LLaMa GPU inference with `llama-cpp` and `cuBLAS`

# %% [markdown]
# ### References
# 
# 1. [YouTube video on GPU inferences](https://www.youtube.com/watch?v=iLBekSpVFq4)
# 2. [GitHub repository with code for GPU inference](https://github.com/MuhammadMoinFaisal/LargeLanguageModelsProjects/blob/main/Run%20Llama2%20Google%20Colab/Llama_2_updated.ipynb)

# %% [markdown]
# ## Import and download dependencies

# %%
%pip install huggingface_hub transformers sentencepiece
!CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --verbose

# %%
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import json

# %% [markdown]
# ## Download model from `huggingface_hub`

# %%
checkpoint = "TheBloke/CodeLlama-13B-Instruct-GGUF"
fileName = r"codellama-13b-instruct.Q4_K_M.gguf"

# %%
modelPath = hf_hub_download(
    repo_id=checkpoint,
    filename=fileName
)

# %% [markdown]
# Here's where the model was downloaded

# %%
modelPath

# %% [markdown]
# ## Load downloaded model

# %%
llm = Llama(
    model_path=modelPath,
    n_threads=2,
    n_batch=512,
    n_gpu_layers=28,
    # n_ctx=3584,
    verbose=True
)

# %% [markdown]
# #### Things to note:
# 
# 1. `n_threads` - refers to your CPU cores
# 2. `n_batches` - needs to be between 1 and `n_ctx`, i.e., the number of characters in the context window. Consider this param when tweaking the code to optimize GPU usage. **Look at how much VRAM you have!**
# 3. `n_gpu_layers` - change this according to the GPU you're using and how much VRAM is has
# 4. If you notice that `BLAS = 1`, this means that `llama-cpp` has setup properly with a GPU backend. In this case, we're usung `cuBLAS` to run inference on an Nvidia GPU.

# %%
with open(r"./test/testScript1.py", "r") as file:
    function = file.read()
    
function

# %%
prompt = f"""Here's my function in Python:

{function}

Given the definition of a function in Python, generate it's documentation. I want it complete with fields like function name, function arguments and return values as well as a detailed explanation of how the function logic works line-by-line. Make it concise and informative to put the documentation into a project."""

# %%
# prompt = f'''SYSTEM: You are a helpful, respectful and honest assistant. With every line of code that you read, try to understand it and explain it's working.

# USER: Write this function's documentation:\n{function}

# ASSISTANT:
# '''

# %%
prompt

# %%
response = llm(
    prompt=prompt, 
    max_tokens=512,
    temperature=0.4, 
    top_p=0.95,
    top_k=150,
    repeat_penalty=1.2, 
    echo=True
)

# %%
print(response)

# %%
print(json.dumps(response, indent=2))

# %%
print(response["choices"][0]["text"])

# %%
import torch, gc

gc.collect()
torch.cuda.empty_cache()


