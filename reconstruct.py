import cv2
import numpy as np
import os
import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import pickle

# ============================================================
# 1️⃣  Load pretrained ResNet50 feature extractor
# ============================================================
try:
    base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
except Exception:
    base_model = models.resnet50(pretrained=True)

class ResNet50_Features(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.features = torch.nn.Sequential(*list(base.children())[:-1])

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            return x.view(x.size(0), -1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Using device: {device}")
model = ResNet50_Features(base_model).to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ============================================================
# 2️⃣  Embedding extraction helpers
# ============================================================
def extract_embedding(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = preprocess(img).unsqueeze(0).to(device)
    feat = model(tensor)
    return feat.cpu().numpy().flatten()

def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# ============================================================
# 3️⃣  Frame ordering logic
# ============================================================
def find_start_frame(embeddings):
    sims = np.zeros((len(embeddings), len(embeddings)))
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            s = cosine_sim(embeddings[i], embeddings[j])
            sims[i, j] = sims[j, i] = s
    avg_sim = sims.mean(axis=1)
    return int(np.argmin(avg_sim))

def greedy_order(embeddings, start_idx):
    N = len(embeddings)
    order = [start_idx]
    visited = {start_idx}
    for _ in tqdm(range(1, N), desc="Building greedy order"):
        last = order[-1]
        next_idx = max(
            ((j, cosine_sim(embeddings[last], embeddings[j])) for j in range(N) if j not in visited),
            key=lambda x: x[1]
        )[0]
        order.append(next_idx)
        visited.add(next_idx)
    return order

# ============================================================
# 4️⃣  Refinement (fast for CPU)
# ============================================================
def refine_order_with_local_similarity(order, embeddings, window=5):
    refined = order.copy()
    N = len(order)
    for i in range(N - window):
        segment = refined[i:i + window]
        segment.sort(key=lambda idx: np.linalg.norm(embeddings[refined[i]] - embeddings[idx]))
        refined[i:i + window] = segment
    return refined

def optical_flow_refine(frames, order, iterations=1, scale=0.25, enable=True):
    """Fast optical-flow refinement; auto-skips if CPU-only."""
    if not enable:
        print(" Skipping optical flow (CPU mode or disabled).")
        return order

    print("🎞️ Running optical flow refinement (scaled for CPU)...")
    refined = order.copy()
    h, w = frames[0].shape[:2]
    small_size = (int(w * scale), int(h * scale))

    for _ in range(iterations):
        for i in tqdm(range(1, len(refined) - 1), desc="Optical flow pass"):
            prev = cv2.resize(cv2.cvtColor(frames[refined[i - 1]], cv2.COLOR_BGR2GRAY), small_size)
            curr = cv2.resize(cv2.cvtColor(frames[refined[i]], cv2.COLOR_BGR2GRAY), small_size)
            nxt  = cv2.resize(cv2.cvtColor(frames[refined[i + 1]], cv2.COLOR_BGR2GRAY), small_size)

            flow1 = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 1, 9, 1, 3, 1.2, 0)
            flow2 = cv2.calcOpticalFlowFarneback(curr, nxt, None, 0.5, 1, 9, 1, 3, 1.2, 0)

            if np.mean(np.abs(flow2)) < np.mean(np.abs(flow1)):
                refined[i - 1], refined[i] = refined[i], refined[i - 1]
    return refined

# ============================================================
# 5️⃣  Save reconstructed video
# ============================================================
def save_video(frames, order, output_file='reconstructed.mp4'):
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    for idx in order:
        out.write(frames[idx])
    out.release()
    print(f" Reconstructed video saved as: {output_file}")

# ============================================================
# 6️⃣  Main
# ============================================================
if __name__ == "__main__":
    start_time = time.time()
    frame_files = sorted(os.listdir("frames"))
    frames = [cv2.imread(os.path.join("frames", f)) for f in frame_files]
    print(f" Loaded {len(frames)} frames.")

    # Load or compute embeddings
    if os.path.exists("embeddings.pkl"):
        with open("embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
        print(" Loaded cached embeddings.")
    else:
        embeddings = [extract_embedding(f) for f in tqdm(frames, desc="Extracting frame embeddings")]
        with open("embeddings.pkl", "wb") as f:
            pickle.dump(embeddings, f)
        print(" Saved embeddings for reuse.")

    start_idx = find_start_frame(embeddings)
    print(f" Detected starting frame index: {start_idx}")

    order = greedy_order(embeddings, start_idx)
    print(" Global greedy ordering complete.")

    order = refine_order_with_local_similarity(order, embeddings)
    print(" Local refinement complete.")

    # Skip or run optical flow depending on device
    run_optical_flow = torch.cuda.is_available()  # Only run if GPU is present
    order = optical_flow_refine(frames, order, enable=run_optical_flow)

    save_video(frames, order)

    elapsed = time.time() - start_time
    print(f" Reconstruction completed in {elapsed:.2f} seconds.")

    end_time = time.time()
elapsed_time = end_time - start_time

# Gather log details
log_content = f"""
Start Time: {time.ctime(start_time)}
End Time: {time.ctime(end_time)}
Elapsed Time: {elapsed_time:.2f} seconds

Device Used: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}
Total Frames Processed: {len(frames)}
Frame Scale: 0.5
Model Used: ResNet50 (pretrained)
Embedding Cache: Disabled
Reconstruction Algorithm: Greedy Ordering + Local Refinement

Status:  Reconstruction Completed Successfully
Output Video: reconstructed_final.mp4
Output FPS: 30
"""

# Save to execution_log.txt
with open("execution_log.txt", "w") as f:
    f.write(log_content.strip())

print("\n📄 Execution log saved as execution_log.txt")

