import pandas as pd
import torch
from transformers import AutoProcessor, SiglipModel
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm
import random
import numpy as np
from sim2score import compute_similarity, similarity_to_score_dict, select_final_answer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

siglip_id = "google/siglip-so400m-patch14-384"
siglip = SiglipModel.from_pretrained(siglip_id).to(device)
siglip_processor = AutoProcessor.from_pretrained(siglip_id)

clip_id = "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
clip = CLIPModel.from_pretrained(clip_id).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_id)

total_siglip = sum(p.numel() for p in siglip.parameters())
total_clip = sum(p.numel() for p in clip.parameters())

print(f"model params: {total_siglip + total_clip:,}")  # 2,244,638,771 ≈ 2.2B

CSV_PATH = "./test.csv"
SUBMISSION_PATH = "sample_submission.csv"

test = pd.read_csv(CSV_PATH)
submission = pd.read_csv(SUBMISSION_PATH)

clip.eval()
siglip.eval()
labels = ["A", "B", "C", "D"]
predictions = []
for idx, row in tqdm(test.iterrows(), total=len(test)):
    image_path = row["img_path"]
    question = row["Question"]
    choices = [row["A"], row["B"], row["C"], row["D"]]
    image = Image.open(image_path).convert("RGB")
    prompts = [f"{choice}" for choice in choices]

    clip_inputs = clip_processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        clip_outputs = clip(**clip_inputs)
        clip_sim = compute_similarity(clip_outputs)
        clip_score_dict = similarity_to_score_dict(clip_sim)

    siglip_inputs = siglip_processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        siglip_outputs = siglip(**siglip_inputs)
        siglip_sim = compute_similarity(siglip_outputs)
        siglip_score_dict = similarity_to_score_dict(siglip_sim)

    prediction = select_final_answer(clip_score_dict, siglip_score_dict)
    predictions.append(prediction)

submission["answer"] = predictions
submission.to_csv("./ensemble_score_0723.csv", index=False)
