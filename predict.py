# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os

import imageio
import numpy as np
import torch
from cog import BasePredictor, Input, Path

import model
import utility
from checkpoint import Checkpoint


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        class Parameters:
            def __init__(self):
                self.cpu = False
                self.n_GPUs = 1
                self.seed = 1
                self.scale = 2
                self.patch_size = 192
                self.rgb_range = 255
                self.n_colors = 3
                self.model = "LBNet-T"
                self.pre_train = "./test_model/LBNet/LBNet-X2.pt"
                self.self_ensemble = True
                self.test_only = True
                self.save = "./tmp/output"

        self.args = Parameters()

        dirs = {
            "LBNet-X2": "LBNet/LBNet-X2.pt",
            "LBNet-X3": "LBNet/LBNet-X3.pt",
            "LBNet-X4": "LBNet/LBNet-X4.pt",
        }

        self.models = {}

        for key in dirs:
            self.models[key] = self.load_model(dirs[key], key)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        utility.set_seed(self.args.seed)

    def predict(
        self,
        image: Path = Input(description="Image to be upscaled"),
        variant: str = Input(
            description="Variant of the model",
            choices=["LBNet-X2", "LBNet-X3", "LBNet-X4"],
            default="LBNet-X2",
        ),
        max_img_height: int = Input(
            description="Maximum image height in pixels (to prevent memory errors)",
            default=400,
        ),
        max_img_width: int = Input(
            description="Maximum image width in pixels (to prevent memory errors)",
            default=400,
        ),
        rgb_range: int = Input(
            description="RGB range of inputted image",
            default=255,
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        img = imageio.imread(image, pilmode="RGB")

        if img.shape[2] != 3:
            raise Exception("Image must have 3 color channels")

        if img.shape[0] > max_img_height:
            raise Exception("Image height is greater than max_img_height")

        if img.shape[1] > max_img_width:
            raise Exception("Image width is greater than max_img_width")

        self.args.rgb_range = rgb_range

        chosen_model = self.models[variant]

        output_dir = "/tmp/output"
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "generated.png")

        with torch.no_grad():
            np_transpose = np.ascontiguousarray(np.transpose(img, [2, 0, 1]))
            tensor = torch.from_numpy(np_transpose).float()
            tensor = torch.unsqueeze(tensor, dim=0)
            tensor = tensor.to(self.device)
            output = chosen_model(tensor)
            output = utility.quantize(output, self.args.rgb_range)

        normalized = output[0]
        ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
        imageio.imwrite(out_path, ndarr)

        return Path(out_path)

    def load_model(self, version, variant):
        model_dir = "./test_model/"
        if "-T" in variant:
            self.args.model = "LBNet-T"
        else:
            self.args.model = "LBNet"
        utility.init_model(self.args)
        print(self.args.model)
        checkpoint = Checkpoint(self.args)
        self.args.pre_train = model_dir + version
        self.args.scale = [int(variant[-1])]
        mod = model.Model(self.args, checkpoint)
        return mod
