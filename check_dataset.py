from pathlib import Path
from PIL import Image

data_dir = Path("style_imgs/512")

image_paths = list(data_dir.glob("*.png")) + list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.jpeg"))

print(f"Found {len(image_paths)} images")

for path in image_paths[:10]:
    img = Image.open(path)
    print(path.name, img.size, img.mode)