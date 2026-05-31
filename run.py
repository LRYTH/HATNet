import os
import subprocess

id_name = "CLIP_HAT_PET_UCM"
cfg = "configs/GeoRSCLIP/georsclip.yml"
gpu_ids = "0"
log_dir = f"log_{id_name}"

# ========================
# 1. 训练
# ========================
train_cmd = [
    "python", "train.py",
    "--cfg", cfg,
    "--id", id_name,
    # ❌ 不建议默认 start_from
    "--gpu_ids", gpu_ids,
]

print("\n[TRAIN]")
print(" ".join(train_cmd))

subprocess.run(train_cmd, check=True)

# ========================
# 2. 评估
# ========================
model_path = f"{log_dir}/model-best.pth"
info_path = f"{log_dir}/infos_{id_name}-best.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(model_path)

if not os.path.exists(info_path):
    raise FileNotFoundError(info_path)

eval_cmd = [
    "python", "eval.py",
    "--dump_json", "1",
    "--dump_images", "0",
    "--num_images", "-1",
    "--model", model_path,
    "--infos_path", info_path,
    "--language_eval", "1",
    "--beam_size", "3"
]

print("\n[EVAL]")
print(" ".join(eval_cmd))

subprocess.run(eval_cmd, check=True)