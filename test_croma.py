import torch
import torch.nn.functional as F
from use_croma import PretrainedCROMA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

S1_PATH = "/Users/dakshanjaria/Kenya_Contrastive_model/outputs/tensors/s1/s1_000000.pt"
S2_PATH = "/Users/dakshanjaria/Kenya_Contrastive_model/outputs/tensors/s2/s2_000000.pt"

MODEL_PATH = "CROMA_base.pt"


def resize_tensor(x, size=120):
    x = x.unsqueeze(0)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x.squeeze(0)


s1 = torch.load(S1_PATH, map_location="cpu").float()
s2 = torch.load(S2_PATH, map_location="cpu").float()

print("Original S1:", s1.shape)
print("Original S2:", s2.shape)

s1 = s1[:2]
s2 = s2[:12]

s1 = resize_tensor(s1, 120)
s2 = resize_tensor(s2, 120)

s1 = s1.unsqueeze(0).to(DEVICE)
s2 = s2.unsqueeze(0).to(DEVICE)

print("CROMA S1:", s1.shape)
print("CROMA S2:", s2.shape)

model = PretrainedCROMA(
    pretrained_path=MODEL_PATH,
    size="base",
    modality="both",
    image_resolution=120,
).to(DEVICE)

model.eval()

with torch.no_grad():
    outputs = model(
        SAR_images=s1,
        optical_images=s2,
    )

print("Available outputs:", outputs.keys())
print("Joint embedding shape:", outputs["joint_GAP"].shape)