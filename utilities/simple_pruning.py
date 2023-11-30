import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from midas.model_loader import load_model

device = torch.device("cpu")
model = load_model(device, "weights/dpt_swin2_large_384.pt", "dpt_swin2_large_384", False, None, False)[0]

# Get number of parameters prior to pruning
num_param_org = sum(p.numel() for p in model.parameters() if p.requires_grad)

# prune the weights
for layers in model.pretrained.model.layers:
  for blocks in layers.blocks:
    prune.l1_unstructured(blocks.attn.qkv, 'weight', amount=0.12)


# remove pruniing renaming of parameters
for layers in model.pretrained.model.layers:
  for blocks in layers.blocks:
    prune.remove(blocks.attn.qkv, 'weight')


# get number of parameters
num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters now pruned {} ({:.8f}%)".format(num_param, num_param/num_param_org))

