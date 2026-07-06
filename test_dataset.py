from src.dataset import StyleImageDataset

dataset = StyleImageDataset(
    data_dir="style_imgs/512",
    instance_token="<sks>",
    resolution=512,
)

print("Number of images:", len(dataset))

sample = dataset[0]

print("Prompt:", sample["prompt"])
print("Image tensor shape:", sample["pixel_values"].shape)
print("Image tensor min:", sample["pixel_values"].min().item())
print("Image tensor max:", sample["pixel_values"].max().item())