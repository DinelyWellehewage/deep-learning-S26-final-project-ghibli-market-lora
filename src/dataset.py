from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class StyleImageDataset(Dataset):
    def __init__(
        self,
        data_dir,
        instance_token,
        resolution=512,
    ):
        self.data_dir = Path(data_dir)
        self.instance_token = instance_token

        self.image_paths = sorted(
            list(self.data_dir.glob("*.png"))
            + list(self.data_dir.glob("*.jpg"))
            + list(self.data_dir.glob("*.jpeg"))
        )

        if len(self.image_paths) == 0:
            raise ValueError(
                f"No images found in {self.data_dir}"
            )

        self.prompt = (
            f"an illustration in {self.instance_token} style"
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    resolution,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return {
            "pixel_values": image,
            "prompt": self.prompt,
        }