import platform
import torch
import torchvision
import torchaudio
import pandas as pd
import transformers
import PIL
import tqdm
import numpy as np

print(platform.system())
print(platform.release())
print(platform.version())
print(platform.platform())

print("PyTorch version:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("GPU name:", torch.cuda.get_device_name(0))
print("Device count:", torch.cuda.device_count())

print("pandas:", pd.__version__)
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("Pillow (PIL):", PIL.__version__)
print("tqdm:", tqdm.__version__)
print("numpy:", np.__version__)
