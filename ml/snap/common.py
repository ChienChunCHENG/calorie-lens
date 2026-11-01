"""Shared paths/helpers for ml/snap scripts."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "web" / "public" / "models"
SP = Path(os.environ.get("SP", "C:/Users/chien/AppData/Local/Temp/claude/C--Users-chien-Downloads-calorie-len/134fdf1a-1063-43e8-b87c-b79446e942b0/scratchpad"))
CLIP_NAME, CLIP_PRETRAINED = "MobileCLIP-S1", "datacompdr"


def load_clip():
    import open_clip, torch
    model, _, pre = open_clip.create_model_and_transforms(CLIP_NAME, pretrained=CLIP_PRETRAINED)
    model.eval()
    from timm.utils import reparameterize_model
    model = reparameterize_model(model)  # fuse the FastViT branches: what we export
    return model, pre, open_clip.get_tokenizer(CLIP_NAME)
