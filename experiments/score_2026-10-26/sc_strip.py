"""pick_e1_L16's model.safetensors holds the AdamW state too; mtstudio score loads weights strictly.
Writes the weights alone (every key not under adamw.) to OUT.safetensors, values untouched.
    sc_strip.py IN.safetensors OUT.safetensors"""
import sys

from safetensors.numpy import load_file, save_file

src, dst = sys.argv[1], sys.argv[2]
t = load_file(src)
keep = {k: v for k, v in t.items() if not k.startswith("adamw.")}
save_file(keep, dst)
print(f"{len(t)} tensors -> {len(keep)} kept ({len(t) - len(keep)} adamw.*)")
