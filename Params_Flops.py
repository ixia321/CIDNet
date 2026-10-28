import torch
from thop import profile
from thop import clever_format
from net.model import make_model

input_tensor = torch.randn(1, 3, 384, 384)

model = make_model("convnext_tiny", num_classes=701)

flops, params = profile(model, inputs=(input_tensor,))
flops, params = clever_format([flops, params], "%.2f")

print(f"Params: {params}")
print(f"FLOPs:  {flops}")
