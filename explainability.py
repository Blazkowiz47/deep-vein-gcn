from logging import StreamHandler, getLogger
from typing import Tuple
import torch
import yaml
from PIL import Image
from numpy.typing import NDArray
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
import numpy as np
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from torchvision import transforms as A
from models import get_model
from cdatasets import get_dataset


log = getLogger("explainability")
log.setLevel("ERROR")
log.addHandler(StreamHandler())


class SimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features

    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=1)
        if len(model_output.shape) == 1:
            model_output = model_output.unsqueeze(0)
        sim = cos(model_output, self.features)
        print(sim)
        return sim


def gradcam(model, target_layers, targets, input_tensor, rgb_img):
    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(
            rgb_img, grayscale_cam, use_rgb=True, image_weight=0.75
        )
        # You can also get the model outputs without having to redo inference
        Image.fromarray(visualization).resize((300, 100)).save("./plots/gradcam.png")


def load_image(fname: str, device: str = "cpu") -> Tuple[torch.Tensor, NDArray]:
    augmentations = A.Compose(
        [
            A.ToTensor(),
            A.RandomHorizontalFlip(p=0.5),
            A.Resize((224, 224)),
        ]
    )
    rgbimage = np.array(Image.open(fname))
    rgbimage = (rgbimage - rgbimage.min()) / (rgbimage.max() - rgbimage.min())
    rgbimage = np.stack([rgbimage, rgbimage, rgbimage], axis=2)
    rgbimage = augmentations(rgbimage)
    imgtensor = rgbimage.unsqueeze(0).float().to(device)
    rgbimage = rgbimage.permute(1, 2, 0).numpy()
    return imgtensor, rgbimage


def driver():
    with open("./configs/dscgrapher2.yaml", "r") as fp:
        config = yaml.safe_load(fp)

    config["leaveoutds"] = "vera"
    wrapper = get_dataset("leaveoneout", config, log)
    trainds = wrapper.get_split("train")
    device = "cpu"
    config["device"] = device
    cids = list(wrapper.train_data.keys())

    if config["num_classes"] != wrapper.num_classes:
        config["num_classes"] = wrapper.num_classes

    ckpt = "dscgrapher_leaveoneout_18_04_25_11_49_2_227"
    model = get_model(config["model"], config, log)
    model.load_state_dict(
        torch.load(f"./tmp/{ckpt}/checkpoints/best_model.pt", map_location=device)
    )
    model.to(device)
    model.eval()

    ds, cid = "fv300", "124"
    classindex = cids.index(f"fv300_{cid}")
    print("Classindex:", classindex)
    enroll_path = f"./data/fv300/1/train/{cid}/6.bmp"

    Image.open(enroll_path).save("./plots/enroll.png")
    enroll_targets = [
        SimilarityToConceptTarget(model(load_image(enroll_path, device)[0]))
    ]
    probe_path = f"./data/fv300/1/train/{cid}/81.bmp"
    Image.open(probe_path).save("./plots/probe.png")

    imgtensor, rgbimage = load_image(probe_path, device)
    print(imgtensor.shape)
    target_layers = [model.backbone[1].grapherblock[-1][-1]]
    print(model.backbone[1].grapherblock[-1][-1])
    gradcam(model, target_layers, enroll_targets, imgtensor, rgbimage)


if __name__ == "__main__":
    driver()
