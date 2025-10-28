python - << 'PY'
import torch, torch_tensorrt, tensorrt as trt
import torch_tensorrt.dynamo as ttrt
print("Torch:", torch.__version__)
print("Torch-TensorRT:", torch_tensorrt.__version__)
print("TensorRT:", trt.__version__)
print("CUDA OK:", torch.cuda.is_available())
print("Dynamo compile OK:", hasattr(ttrt, "compile"))
PY
