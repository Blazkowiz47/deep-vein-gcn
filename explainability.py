from logging import StreamHandler, getLogger
import random
from tqdm import tqdm
import os
import json
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


def gradcam(
    model, target_layers, targets, input_tensor, rgb_img, fname="./plots/gradcam.png"
):
    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(
            rgb_img, grayscale_cam, use_rgb=True, image_weight=0.7
        )
        # You can also get the model outputs without having to redo inference
        Image.fromarray(visualization).save(fname)


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
    rgbimage = rgbimage.permute(1, 2, 0).numpy().astype(np.float32).clip(0, 1)
    return imgtensor, rgbimage


def plot_gradcam_and_enrol_probe_v2(
    enroll_path: str,
    probe_path: str,
    gradcam_path: str,
    model,
):
    enroll_targets = [SimilarityToConceptTarget(model(load_image(enroll_path)[0]))]

    imgtensor, rgbimage = load_image(probe_path)
    target_layers = [model.stem[-1]]
    gradcam(
        model,
        target_layers,
        enroll_targets,
        imgtensor,
        rgbimage,
        gradcam_path,
    )


def plot_gradcam_and_enrol_probe(
    enroll_path: str,
    probe_path: str,
    ofenroll: str,
    ofprobe: str,
    gradcam_path: str,
    model,
):
    Image.open(enroll_path).resize((224, 224)).save(ofenroll)
    enroll_targets = [SimilarityToConceptTarget(model(load_image(enroll_path)[0]))]

    Image.open(probe_path).resize((224, 224)).save(ofprobe)

    imgtensor, rgbimage = load_image(probe_path)
    target_layers = [model.backbone[1].grapherblock[-1][-1]]
    gradcam(
        model,
        target_layers,
        enroll_targets,
        imgtensor,
        rgbimage,
        gradcam_path,
    )


def driver():
    with open("./configs/dscgrapher2.yaml", "r") as fp:
        config = yaml.safe_load(fp)

    device = "cuda"
    config["leaveoutds"] = "vera"
    wrapper = get_dataset("leaveoneout", config, log)
    trainds = wrapper.get_split("train", return_filename=True)
    testds = wrapper.get_split("test", return_filename=True)
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
    dataset_features = {}
    for dataset in [trainds, testds]:
        for imgs, _, fnames in tqdm(dataset):
            imgs = imgs.to(device)
            feats = model(imgs)
            feats = feats.cpu().detach().numpy().tolist()
            fnames = fnames
            for fname, feat in zip(fnames, feats):
                _, _, ds, *_, cid, _ = fname.split("/")
                if ds not in dataset_features:
                    dataset_features[ds] = {}

                cid = ds + "_" + cid
                if cid not in dataset_features[ds]:
                    dataset_features[ds][cid] = []

                dataset_features[ds][cid].append((feat, fname))

    config["device"] = "cpu"
    model = get_model(config["model"], config, log)
    model.load_state_dict(
        torch.load(f"./tmp/{ckpt}/checkpoints/best_model.pt", map_location=device)
    )
    for ds in dataset_features:
        class_features = dataset_features[ds]
        cids = list(class_features.keys())
        cid1 = random.choice(cids)
        cid2 = random.choice([c for c in cids if c != cid1])
        for i, feat in enumerate(class_features[cid1]):
            class_features[cid1][i] = (torch.tensor(feat[0]).unsqueeze(dim=0), feat[1])
        for i, feat in enumerate(class_features[cid2]):
            class_features[cid2][i] = (torch.tensor(feat[0]).unsqueeze(dim=0), feat[1])

        genuine_scores = []
        imposter_scores = []
        for feat1, fname1 in class_features[cid1]:
            for feat2, fname2 in class_features[cid1]:
                if fname1 == fname2:
                    continue
                genuine_scores.append(
                    (
                        torch.nn.functional.cosine_similarity(
                            feat1, feat2, dim=1
                        ).item(),
                        fname1,
                        fname2,
                    )
                )

        for feat1, fname1 in class_features[cid2]:
            for feat2, fname2 in class_features[cid2]:
                if fname1 == fname2:
                    continue
                genuine_scores.append(
                    (
                        torch.nn.functional.cosine_similarity(
                            feat1, feat2, dim=1
                        ).item(),
                        fname1,
                        fname2,
                    )
                )

        genuine_scores.sort(key=lambda x: x[0], reverse=True)

        for feat1, fname1 in class_features[cid1]:
            for feat2, fname2 in class_features[cid2]:
                imposter_scores.append(
                    (
                        torch.nn.functional.cosine_similarity(feat1, feat2).item(),
                        fname1,
                        fname2,
                    )
                )

        imposter_scores.sort(key=lambda x: x[0], reverse=True)
        print(
            f"Genuine scores for {cid1} and {cid2}: {genuine_scores[0]}, {genuine_scores[-1]}"
        )
        print(
            f"Imposter scores for {cid1} and {cid2}: {imposter_scores[0]}, {imposter_scores[-1]}"
        )

        # Genuine max case
        plot_gradcam_and_enrol_probe(
            genuine_scores[0][1],
            genuine_scores[0][2],
            f"./plots/{ds}_gen_max_enroll.png",
            f"./plots/{ds}_gen_max_probe.png",
            f"./plots/{ds}_gen_max_grad.png",
            model,
        )

        # Genuine min case
        plot_gradcam_and_enrol_probe(
            genuine_scores[-1][1],
            genuine_scores[-1][2],
            f"./plots/{ds}_gen_min_enroll.png",
            f"./plots/{ds}_gen_min_probe.png",
            f"./plots/{ds}_gen_min_grad.png",
            model,
        )

        # Imposter max case
        plot_gradcam_and_enrol_probe(
            imposter_scores[0][1],
            imposter_scores[0][2],
            f"./plots/{ds}_imp_max_enroll.png",
            f"./plots/{ds}_imp_max_probe.png",
            f"./plots/{ds}_imp_max_grad.png",
            model,
        )

        # Imposter min case
        plot_gradcam_and_enrol_probe(
            imposter_scores[-1][1],
            imposter_scores[-1][2],
            f"./plots/{ds}_imp_min_enroll.png",
            f"./plots/{ds}_imp_min_probe.png",
            f"./plots/{ds}_imp_min_grad.png",
            model,
        )


def driver2():
    with open("./configs/dscgrapher2.yaml", "r") as fp:
        config = yaml.safe_load(fp)

    device = "cpu"
    config["leaveoutds"] = "vera"
    wrapper = get_dataset("leaveoneout", config, log)
    trainds = wrapper.get_split("train", return_filename=True)
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
    print(model)
    return
    for ds in tqdm(["polyu", "fv300", "fvusm",  "mmcbnu"]):
        enrol = f"./plots/{ds}_gen_min_enroll.png"
        probe = f"./plots/{ds}_gen_min_probe.png"
        grad = f"./plots/{ds}_gen_min_grad_stem.png"
        plot_gradcam_and_enrol_probe_v2(enrol, probe, grad, model)

        enrol = f"./plots/{ds}_gen_max_enroll.png"
        probe = f"./plots/{ds}_gen_max_probe.png"
        grad = f"./plots/{ds}_gen_max_grad_stem.png"
        plot_gradcam_and_enrol_probe_v2(enrol, probe, grad, model)

        enrol = f"./plots/{ds}_imp_min_enroll.png"
        probe = f"./plots/{ds}_imp_min_probe.png"
        grad = f"./plots/{ds}_imp_min_grad_stem.png"
        plot_gradcam_and_enrol_probe_v2(enrol, probe, grad, model)

        enrol = f"./plots/{ds}_imp_max_enroll.png"
        probe = f"./plots/{ds}_imp_max_probe.png"
        grad = f"./plots/{ds}_imp_max_grad_stem.png"
        plot_gradcam_and_enrol_probe_v2(enrol, probe, grad, model)


if __name__ == "__main__":
    driver2()

