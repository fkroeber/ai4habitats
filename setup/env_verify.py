"""GPU sanity test for the ai4habitats environment.

Run inside an srun GPU session if necessary, e.g.:
    srun --partition=mlgpu_devel --time=00:30:00 --gpus=1 --ntasks=1 --pty bash
    conda activate ai4habitats
    python env_verify.py
"""

import subprocess
import sys

print("=" * 60)
print("driver check")
print("=" * 60)
subprocess.run(
    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"]
)

import torch  # noqa: E402  (torch first, matching your workflow)
import tensorflow as tf  # noqa: E402

print("=" * 60)
print(f"torch {torch.__version__} | tf {tf.__version__}")
print("=" * 60)

ok = True

# --- torch ---
if torch.cuda.is_available():
    dev = torch.cuda.get_device_name(0)
    a = torch.randn(2048, 2048, device="cuda")
    r = (a @ a).sum().item()
    print(f"[OK] torch GPU: {dev}, matmul sum = {r:.1f}")
    print(f"[OK] torch cuDNN runtime: {torch.backends.cudnn.version()}")
else:
    ok = False
    print("[FAIL] torch.cuda.is_available() is False")

# --- tensorflow ---
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"[OK] tf sees {len(gpus)} GPU(s): {gpus}")
    with tf.device("/GPU:0"):
        x = tf.random.normal((2048, 2048))
        r = tf.reduce_sum(tf.matmul(x, x)).numpy()
    print(f"[OK] tf GPU matmul sum = {r:.1f}")
    # exercise cuDNN explicitly (conv), the layer that used to break
    with tf.device("/GPU:0"):
        img = tf.random.normal((1, 64, 64, 3))
        out = tf.keras.layers.Conv2D(8, 3)(img)
    print(f"[OK] tf cuDNN conv output shape = {out.shape}")
else:
    ok = False
    print("[FAIL] tf.config.list_physical_devices('GPU') is empty")
    print(
        "       -> check: pip show nvidia-cudnn-cu12   (must match the version pinned by tensorflow[and-cuda]==2.16.2, see your lock file)"
    )
    print(
        "       -> check: echo $LD_LIBRARY_PATH        (should not point to system/conda CUDA)"
    )

# --- both frameworks on GPU in the same process ---
if ok:
    t = torch.ones(512, 512, device="cuda").sum().item()
    f = tf.reduce_sum(tf.ones((512, 512))).numpy()
    print(f"[OK] coexistence: torch={t:.0f}, tf={f:.0f} in one process")
    print("\nALL CHECKS PASSED")
else:
    sys.exit(1)
