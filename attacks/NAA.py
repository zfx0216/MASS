# NAA
# Fill every "......" path and review the configuration before running.
# See README.md and docs/EXPERIMENTS.md for the evaluated protocol.
# Reference: https://github.com/jpzhang1810/NAA
# Reference: https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Improving_Adversarial_Transferability_via_Neuron_Attribution-Based_Attacks_CVPR_2022_paper.html

from pathlib import Path
import csv
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

MODEL_PATH = "......"
MODEL_NAME = "resnet50"
MODEL_BACKEND = "torchvision"
CLASS_INDEX_PATH = "......"
INPUT_IMAGE_DIR = "......"
OUTPUT_IMAGE_DIR = "......"
MANIFEST_PATH = "......"
DEVICE = "cuda:0"
BATCH_SIZE = 16
NUM_WORKERS = 4
PIN_MEMORY = True
EPS = 16 / 255
STEPS = 10
ALPHA = EPS / STEPS
MOMENTUM = 1.0
INTEGRATION_STEPS = 30
ATTRIBUTION_GAMMA = 1.0
INTEGRATION_MODE = "paper"
AUTHOR_PATH_NOISE_STD = 0.2
FEATURE_LAYER = "auto"
FEATURE_LAYER_MAP = {
    "resnet18": "layer2",
    "resnet50": "layer2",
    "vgg16": "features.16",
    "densenet121": "features.denseblock2",
    "mobilenet_v2": "features.6",
}
TIMM_CONVNEXT_LAYER = "stages.2"
TIMM_VIT_LAYER = "blocks.6"
NUM_CLASSES = 1000
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
OVERWRITE = False


def load_class_index() -> dict[int, str]:
    with Path(CLASS_INDEX_PATH).open("r", encoding="utf-8-sig") as handle:
        entries = json.load(handle)
    if set(entries) != {str(i) for i in range(NUM_CLASSES)}:
        raise ValueError("The class index must contain every key from 0 through NUM_CLASSES-1.")
    mapping: dict[int, str] = {}
    for index, entry in entries.items():
        if isinstance(entry, str) and entry.strip():
            name = entry
        elif isinstance(entry, list) and len(entry) == 2 and isinstance(entry[1], str):
            name = entry[1]
        else:
            raise ValueError(f"Invalid class-index entry: {index}: {entry}")
        mapping[int(index)] = name
    return mapping


def load_model(device: torch.device) -> nn.Module:
    if MODEL_BACKEND == "torchvision":
        constructors = {
            "resnet18": models.resnet18,
            "resnet50": models.resnet50,
            "vgg16": models.vgg16,
            "densenet121": models.densenet121,
            "mobilenet_v2": models.mobilenet_v2,
        }
        if MODEL_NAME not in constructors:
            raise ValueError(f"Unsupported torchvision model: {MODEL_NAME}")
        model = constructors[MODEL_NAME](weights=None, num_classes=NUM_CLASSES)
    elif MODEL_BACKEND == "timm":
        import timm

        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    else:
        raise ValueError("MODEL_BACKEND must be torchvision or timm.")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a state_dict or a dictionary containing one.")
    state = checkpoint
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(checkpoint.get(key), dict):
            state = checkpoint[key]
            break
    for prefix in ("module.", "_orig_mod.", "module."):
        if state and all((key.startswith(prefix) for key in state)):
            state = {key[len(prefix) :]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    model.requires_grad_(False)
    return model


class ImageFolderDataset(Dataset):

    def __init__(self):
        self.root = Path(INPUT_IMAGE_DIR).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {self.root}")
        self.paths = sorted(
            (
                p
                for p in self.root.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
        )
        if not self.paths:
            raise ValueError(f"No images in input directory: {self.root}")
        self.names: list[str] = []
        seen: set[str] = set()
        for path in self.paths:
            relative = path.relative_to(self.root)
            name = relative.as_posix()
            output_name = relative.with_suffix(".png").as_posix()
            if output_name.casefold() in seen:
                raise ValueError(f"Output filename collision: {output_name}")
            seen.add(output_name.casefold())
            self.names.append(name)
        self.transform = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(
                    f"Image size must be {IMAGE_SIZE}x{IMAGE_SIZE}:{self.paths[index]}; actual size {image.size}"
                )
            pixels = self.transform(image)
        return (pixels, self.names[index])


def normalize(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def attack_parameters() -> dict:
    return {
        "feature_layer": feature_layer_name(),
        "integration_steps": INTEGRATION_STEPS,
        "integration_mode": INTEGRATION_MODE,
        "attribution_gamma": ATTRIBUTION_GAMMA,
        "author_path_noise_std": AUTHOR_PATH_NOISE_STD if INTEGRATION_MODE == "author" else 0.0,
    }


def unit_gradient(gradient: torch.Tensor, mean: bool = False) -> torch.Tensor:
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("The gradient contains NaN/Inf; check the model and parameters.")
    magnitude = (
        gradient.abs().mean((1, 2, 3), keepdim=True)
        if mean
        else gradient.abs().sum((1, 2, 3), keepdim=True)
    )
    return gradient / torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))


def feature_layer_name() -> str:
    if FEATURE_LAYER != "auto":
        return FEATURE_LAYER
    if MODEL_BACKEND == "torchvision" and MODEL_NAME in FEATURE_LAYER_MAP:
        return FEATURE_LAYER_MAP[MODEL_NAME]
    if MODEL_BACKEND == "timm" and MODEL_NAME.startswith("convnext"):
        return TIMM_CONVNEXT_LAYER
    if MODEL_BACKEND == "timm" and MODEL_NAME.startswith("vit"):
        return TIMM_VIT_LAYER
    raise ValueError("Unknown NAA feature layer; set FEATURE_LAYER explicitly.")


def naa(model: nn.Module, clean: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    original = clean.detach()
    baseline = torch.zeros_like(original)
    layer_name = feature_layer_name()
    modules = dict(model.named_modules())
    if layer_name not in modules:
        raise ValueError(f"Model has no feature layer {layer_name}; check FEATURE_LAYER.")
    captured: dict[str, torch.Tensor] = {}

    def capture(_module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError("The NAA feature layer must output one Tensor.")
        captured["feature"] = output

    inplace_modules = [m for m in model.modules() if getattr(m, "inplace", False)]
    for module in inplace_modules:
        module.inplace = False
    hook = modules[layer_name].register_forward_hook(capture)
    try:
        integrated_attention = None
        for m in range(1, INTEGRATION_STEPS + 1):
            with torch.enable_grad():
                if INTEGRATION_MODE == "paper":
                    virtual = (
                        baseline + m / INTEGRATION_STEPS * (original - baseline)
                    ).requires_grad_(True)
                    logits = model(normalize(virtual))
                else:
                    fraction = (m - 1) / INTEGRATION_STEPS
                    virtual = (
                        (
                            (
                                normalize(original)
                                + torch.randn_like(original) * AUTHOR_PATH_NOISE_STD
                            )
                            * (1 - fraction)
                            + normalize(baseline) * fraction
                        )
                        .detach()
                        .requires_grad_(True)
                    )
                    logits = model(virtual)
                feature = captured.pop("feature")
                probability = logits.softmax(dim=1).gather(1, labels[:, None]).sum()
                attention = torch.autograd.grad(probability, feature)[0].detach()
            if not torch.isfinite(attention).all():
                raise FloatingPointError("NAA integrated attention contains NaN/Inf.")
            if integrated_attention is None:
                integrated_attention = torch.zeros_like(attention)
            integrated_attention += attention / INTEGRATION_STEPS
        with torch.no_grad():
            model(normalize(baseline))
            baseline_feature = captured.pop("feature").detach()
        adversarial = original.clone()
        momentum = torch.zeros_like(original)
        lower, upper = ((original - EPS).clamp(0, 1), (original + EPS).clamp(0, 1))
        for _ in range(STEPS):
            with torch.enable_grad():
                point = adversarial.detach().requires_grad_(True)
                model(normalize(point))
                feature = captured.pop("feature")
                attribution = (feature - baseline_feature) * integrated_attention
                weighted = torch.where(
                    attribution >= 0, attribution, ATTRIBUTION_GAMMA * attribution
                )
                gradient = torch.autograd.grad(weighted.sum(), point)[0].detach()
            momentum = MOMENTUM * momentum + unit_gradient(gradient)
            adversarial = point.detach() - ALPHA * momentum.sign()
            adversarial = torch.maximum(lower, torch.minimum(upper, adversarial))
        return adversarial.detach()
    finally:
        hook.remove()
        captured.clear()
        for module in inplace_modules:
            module.inplace = True


def quantize_images(clean: torch.Tensor, adversarial: torch.Tensor) -> torch.Tensor:
    clean_u8 = (clean.detach().cpu() * 255).round().clamp(0, 255)
    adv = adversarial.detach().cpu() * 255
    lower = (clean_u8 - EPS * 255).ceil().clamp(0, 255)
    upper = (clean_u8 + EPS * 255).floor().clamp(0, 255)
    adv_u8 = torch.maximum(lower, torch.minimum(upper, adv.round()))
    return adv_u8.to(torch.uint8)


def save_png(pixels: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = pixels.permute(1, 2, 0).contiguous().numpy()
    Image.fromarray(array).save(path, format="PNG")


def _check_paths_configured() -> None:
    """Reject unresolved path placeholders before any model or output is used."""
    missing = []

    def inspect(value, name):
        if isinstance(value, str) and value == "......":
            missing.append(name)
        elif isinstance(value, dict):
            for key, child in value.items():
                inspect(child, f"{name}[{key!r}]")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                inspect(child, f"{name}[{index}]")

    for name, value in globals().items():
        if name.isupper():
            inspect(value, name)
    if missing:
        raise ValueError(
            "Replace the '......' path placeholders before running: " + ", ".join(missing)
        )


def main() -> None:
    _check_paths_configured()
    if not 0 <= EPS <= 1 or BATCH_SIZE <= 0 or NUM_WORKERS < 0:
        raise ValueError("Require 0 <= EPS <= 1, BATCH_SIZE > 0, and NUM_WORKERS >= 0.")
    if not isinstance(STEPS, int) or isinstance(STEPS, bool) or STEPS <= 0:
        raise ValueError("STEPS must be a positive integer.")
    if not 0 <= ALPHA <= 1 or (EPS > 0 and ALPHA == 0):
        raise ValueError("Require 0 <= ALPHA <= 1, with ALPHA > 0 when EPS > 0.")
    if not 0 <= MOMENTUM < float("inf"):
        raise ValueError("MOMENTUM must be finite and nonnegative.")
    if IMAGE_SIZE <= 0:
        raise ValueError("Require IMAGE_SIZE > 0.")
    if not isinstance(INTEGRATION_STEPS, int) or INTEGRATION_STEPS <= 0:
        raise ValueError("INTEGRATION_STEPS must be a positive integer.")
    if INTEGRATION_MODE not in ("paper", "author"):
        raise ValueError("INTEGRATION_MODE must be paper or author.")
    if not 0 <= ATTRIBUTION_GAMMA < float("inf") or not 0 <= AUTHOR_PATH_NOISE_STD < float("inf"):
        raise ValueError(
            "Attribution weight and integration noise standard deviation must be finite and nonnegative."
        )
    device = torch.device(DEVICE)
    if device.type == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError(
            "CUDA is unavailable; install a CUDA build of PyTorch or set DEVICE to cpu."
        )
    class_names = load_class_index()
    dataset = ImageFolderDataset()
    output_root = Path(OUTPUT_IMAGE_DIR).resolve()
    manifest_path = Path(MANIFEST_PATH).resolve()
    if output_root == dataset.root or dataset.root in output_root.parents:
        raise ValueError("The output directory must be outside the input directory.")
    if not OVERWRITE:
        if manifest_path.exists():
            raise FileExistsError(f"Manifest already exists: {manifest_path}")
        for name in dataset.names:
            path = output_root / Path(name).with_suffix(".png")
            if path.exists():
                raise FileExistsError(
                    f"File already exists: {path}; choose another output directory or enable OVERWRITE."
                )
    model = load_model(device)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and device.type == "cuda",
        drop_last=False,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    successful_images = 0
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "row_type",
            "filename",
            "generated_filename",
            "attack_label",
            "attack_class_name",
            "label_source",
            "clean_prediction",
            "adversarial_prediction",
            "adversarial_class_name",
            "method",
            "source_model",
            "eps",
            "steps",
            "alpha",
            "momentum",
            "attack_success",
            "successful_images",
            "total_images",
            "ASR_percent",
        ]
        fields += list(attack_parameters())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for clean, names in loader:
            clean = clean.to(device, non_blocking=True)
            with torch.no_grad():
                clean_predictions = model(normalize(clean)).argmax(dim=1)
            attack_labels = clean_predictions
            adversarial = naa(model, clean, attack_labels)
            adv_u8 = quantize_images(clean, adversarial)
            with torch.no_grad():
                saved_pixels = adv_u8.to(device=device, dtype=clean.dtype) / 255
                adv_predictions = model(normalize(saved_pixels)).argmax(dim=1)
            batch_labels = attack_labels.cpu().tolist()
            clean_preds = clean_predictions.cpu().tolist()
            adv_preds = adv_predictions.cpu().tolist()
            for i, name in enumerate(names):
                relative_output = Path(name).with_suffix(".png").as_posix()
                save_png(adv_u8[i], output_root / relative_output)
                attack_success = int(adv_preds[i] != clean_preds[i])
                successful_images += attack_success
                completed += 1
                writer.writerow(
                    {
                        "row_type": "image",
                        "filename": name,
                        "generated_filename": relative_output,
                        "attack_label": batch_labels[i],
                        "attack_class_name": class_names[batch_labels[i]],
                        "label_source": "source_model_prediction",
                        "clean_prediction": clean_preds[i],
                        "adversarial_prediction": adv_preds[i],
                        "adversarial_class_name": class_names[adv_preds[i]],
                        "method": "NAA",
                        "source_model": MODEL_NAME,
                        "eps": EPS,
                        "steps": STEPS,
                        "alpha": ALPHA,
                        "momentum": MOMENTUM,
                        **attack_parameters(),
                        "attack_success": attack_success,
                        "successful_images": successful_images,
                        "total_images": completed,
                        "ASR_percent": f"{100 * successful_images / completed:.6f}",
                    }
                )
            handle.flush()
            print(
                f"NAA: processed {completed}/{len(dataset)} images; cumulative success rate {100 * successful_images / completed:.2f}%"
            )
        writer.writerow(
            {
                "row_type": "summary",
                "label_source": "source_model_prediction",
                "method": "NAA",
                "source_model": MODEL_NAME,
                "eps": EPS,
                "steps": STEPS,
                "alpha": ALPHA,
                "momentum": MOMENTUM,
                **attack_parameters(),
                "successful_images": successful_images,
                "total_images": completed,
                "ASR_percent": f"{100 * successful_images / completed:.6f}",
            }
        )
    print(f"Final source prediction-change rate: {100 * successful_images / completed:.2f}%")
    print(f"Generated images: {output_root}\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
