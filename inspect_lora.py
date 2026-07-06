from safetensors.torch import load_file

path = "lora_out/pytorch_lora_weights.safetensors"

state_dict = load_file(path)

print("Number of keys:", len(state_dict))
print("\nFirst 30 keys:")

for i, key in enumerate(state_dict.keys()):
    if i >= 30:
        break
    print(key, state_dict[key].shape)