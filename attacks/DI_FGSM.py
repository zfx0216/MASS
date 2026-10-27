# DI_FGSM
# Fill every "......" path and review the configuration before running.
# See README.md and docs/EXPERIMENTS.md for the evaluated protocol.
# Reference: https://arxiv.org/abs/1803.06978
# Reference: https://github.com/cihangxie/DI-2-FGSM

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
MOMENTUM = 0.0
DIVERSITY_PROB = 0.5
DIVERSITY_MIN_SIZE = 224
DIVERSITY_MAX_SIZE = 248
DIVERSITY_PAD_VALUE = 0.0
DIVERSITY_INTERPOLATION = "nearest"
DIVERSITY_RESIZE_BACK = False
METHOD_NAME = "DI_FGSM" if MOMENTUM == 0 else "M_DI_FGSM"
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


def input_diversity(normalized: torch.Tensor) -> torch.Tensor:
    if DIVERSITY_PROB == 0:
        return normalized
    if torch.rand((), device=normalized.device).item() >= DIVERSITY_PROB:
        return normalized
    size = int(
        torch.randint(DIVERSITY_MIN_SIZE, DIVERSITY_MAX_SIZE, (), device=normalized.device).item()
    )
    resized = F.interpolate(normalized, size=(size, size), mode=DIVERSITY_INTERPOLATION)
    remaining = DIVERSITY_MAX_SIZE - size
    top = int(torch.randint(remaining, (), device=normalized.device).item())
    left = int(torch.randint(remaining, (), device=normalized.device).item())
    transformed = F.pad(
        resized,
        (left, remaining - left, top, remaining - top),
        mode="constant",
        value=DIVERSITY_PAD_VALUE,
    )
    if DIVERSITY_RESIZE_BACK:
        transformed = F.interpolate(
            transformed, size=(IMAGE_SIZE, IMAGE_SIZE), mode=DIVERSITY_INTERPOLATION
        )
    return transformed


def di_fgsm(model: nn.Module, clean: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    original = clean.detach()
    adversarial = original.clone()
    momentum = torch.zeros_like(original)
    lower = (original - EPS).clamp(0, 1)
    upper = (original + EPS).clamp(0, 1)
    for _ in range(STEPS):
        adversarial = adversarial.detach().requires_grad_(True)
        with torch.enable_grad():
            logits = model(input_diversity(normalize(adversarial)))
            loss = F.cross_entropy(logits, labels, reduction="sum")
            gradient = torch.autograd.grad(loss, adversarial, only_inputs=True)[0]
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(
                "The input gradient contains NaN/Inf; check the weights and inputs."
            )
        with torch.no_grad():
            norm = gradient.abs().sum(dim=(1, 2, 3), keepdim=True)
            denominator = torch.where(norm > 0, norm, torch.ones_like(norm))
            momentum = MOMENTUM * momentum + gradient / denominator
            adversarial = adversarial + ALPHA * momentum.sign()
            adversarial = torch.maximum(lower, torch.minimum(upper, adversarial))
    return adversarial.detach()


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
    if not 0 <= DIVERSITY_PROB <= 1:
        raise ValueError("DIVERSITY_PROB must be in [0,1].")
    if not 0 < DIVERSITY_MIN_SIZE < DIVERSITY_MAX_SIZE:
        raise ValueError("Require 0 < DIVERSITY_MIN_SIZE < DIVERSITY_MAX_SIZE.")
    if DIVERSITY_INTERPOLATION != "nearest":
        raise ValueError("This implementation requires nearest-neighbor interpolation.")
    if IMAGE_SIZE <= 0:
        raise ValueError("Require IMAGE_SIZE > 0.")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
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
    patch_embed = getattr(model, "patch_embed", None)
    if (
        DIVERSITY_PROB > 0
        and (not DIVERSITY_RESIZE_BACK)
        and getattr(patch_embed, "strict_img_size", False)
    ):
        raise ValueError("This model requires a fixed input size; set DIVERSITY_RESIZE_BACK=True.")
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
            "diversity_prob",
            "diversity_min_size",
            "diversity_max_size",
            "diversity_resize_back",
            "diversity_pad_value",
            "diversity_interpolation",
            "attack_success",
            "successful_images",
            "total_images",
            "ASR_percent",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for clean, names in loader:
            clean = clean.to(device, non_blocking=True)
            with torch.no_grad():
                clean_predictions = model(normalize(clean)).argmax(dim=1)
            attack_labels = clean_predictions
            adversarial = di_fgsm(model, clean, attack_labels)
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
                        "method": METHOD_NAME,
                        "source_model": MODEL_NAME,
                        "eps": EPS,
                        "steps": STEPS,
                        "alpha": ALPHA,
                        "momentum": MOMENTUM,
                        "diversity_prob": DIVERSITY_PROB,
                        "diversity_min_size": DIVERSITY_MIN_SIZE,
                        "diversity_max_size": DIVERSITY_MAX_SIZE,
                        "diversity_resize_back": DIVERSITY_RESIZE_BACK,
                        "diversity_pad_value": DIVERSITY_PAD_VALUE,
                        "diversity_interpolation": DIVERSITY_INTERPOLATION,
                        "attack_success": attack_success,
                        "successful_images": successful_images,
                        "total_images": completed,
                        "ASR_percent": f"{100 * successful_images / completed:.6f}",
                    }
                )
            handle.flush()
            print(
                f"{METHOD_NAME}: processed {completed}/{len(dataset)} images; cumulative success rate {100 * successful_images / completed:.2f}%"
            )
        writer.writerow(
            {
                "row_type": "summary",
                "label_source": "source_model_prediction",
                "method": METHOD_NAME,
                "source_model": MODEL_NAME,
                "eps": EPS,
                "steps": STEPS,
                "alpha": ALPHA,
                "momentum": MOMENTUM,
                "diversity_prob": DIVERSITY_PROB,
                "diversity_min_size": DIVERSITY_MIN_SIZE,
                "diversity_max_size": DIVERSITY_MAX_SIZE,
                "diversity_resize_back": DIVERSITY_RESIZE_BACK,
                "diversity_pad_value": DIVERSITY_PAD_VALUE,
                "diversity_interpolation": DIVERSITY_INTERPOLATION,
                "successful_images": successful_images,
                "total_images": completed,
                "ASR_percent": f"{100 * successful_images / completed:.6f}",
            }
        )
    print(f"Final source prediction-change rate: {100 * successful_images / completed:.2f}%")
    print(f"Generated images: {output_root}\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
