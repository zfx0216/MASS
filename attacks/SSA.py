# SSA
# Fill every "......" path and review the configuration before running.
# See README.md and docs/EXPERIMENTS.md for the evaluated protocol.
# Reference: https://arxiv.org/abs/2207.05382
# Reference: https://github.com/yuyang-long/SSA

from pathlib import Path
import csv
import json
import math
import time
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
METHOD_NAME = "SSA"
DEVICE = "cuda:0"
BATCH_SIZE = 16
NUM_WORKERS = 4
PIN_MEMORY = True
EPS = 16 / 255
STEPS = 10
ALPHA = EPS / STEPS
SPECTRUM_SAMPLES = 20
RHO = 0.5
SIGMA = EPS
GRADIENT_MODE = "author"
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


def normalize(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


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


def dct_1d(pixels: torch.Tensor) -> torch.Tensor:
    shape = pixels.shape
    size = shape[-1]
    flat = pixels.contiguous().view(-1, size)
    reordered = torch.cat((flat[:, ::2], flat[:, 1::2].flip([1])), dim=1)
    spectrum = torch.fft.fft(reordered, dim=1)
    phase = (
        -torch.arange(size, dtype=pixels.dtype, device=pixels.device)[None, :]
        * math.pi
        / (2 * size)
    )
    transformed = spectrum.real * phase.cos() - spectrum.imag * phase.sin()
    return (2 * transformed).view(shape)


def idct_1d(coefficients: torch.Tensor) -> torch.Tensor:
    shape = coefficients.shape
    size = shape[-1]
    real = coefficients.contiguous().view(-1, size) / 2
    imag = torch.cat((real[:, :1] * 0, -real.flip([1])[:, :-1]), dim=1)
    phase = (
        torch.arange(size, dtype=coefficients.dtype, device=coefficients.device)[None, :]
        * math.pi
        / (2 * size)
    )
    spectrum = torch.complex(
        real * phase.cos() - imag * phase.sin(), real * phase.sin() + imag * phase.cos()
    )
    reordered = torch.fft.ifft(spectrum, dim=1).real
    pixels = torch.empty_like(reordered)
    pixels[:, ::2] = reordered[:, : size - size // 2]
    pixels[:, 1::2] = reordered.flip([1])[:, : size // 2]
    return pixels.view(shape)


def dct_2d(pixels: torch.Tensor) -> torch.Tensor:
    return dct_1d(dct_1d(pixels).transpose(-1, -2)).transpose(-1, -2)


def idct_2d(coefficients: torch.Tensor) -> torch.Tensor:
    return idct_1d(idct_1d(coefficients).transpose(-1, -2)).transpose(-1, -2)


def spectrum_transform(pixels: torch.Tensor) -> torch.Tensor:
    noisy = pixels + torch.randn_like(pixels) * SIGMA
    coefficients = dct_2d(noisy)
    mask = torch.rand_like(pixels) * (2 * RHO) + (1 - RHO)
    return idct_2d(coefficients * mask)


def ssa(model: nn.Module, clean: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    original = clean.detach()
    adversarial = original.clone()
    lower = (original - EPS).clamp(0, 1)
    upper = (original + EPS).clamp(0, 1)
    for _ in range(STEPS):
        average = torch.zeros_like(original)
        for _ in range(SPECTRUM_SAMPLES):
            with torch.enable_grad():
                if GRADIENT_MODE == "author":
                    with torch.no_grad():
                        transformed = spectrum_transform(adversarial)
                    point = transformed.detach().requires_grad_(True)
                    logits = model(normalize(point))
                else:
                    point = adversarial.detach().requires_grad_(True)
                    logits = model(normalize(spectrum_transform(point)))
                loss = F.cross_entropy(logits, labels, reduction="sum")
                gradient = torch.autograd.grad(loss, point)[0].detach()
            if not torch.isfinite(gradient).all():
                raise FloatingPointError("SSA gradients contain NaN/Inf.")
            average += gradient
        average /= SPECTRUM_SAMPLES
        adversarial = adversarial + ALPHA * average.sign()
        adversarial = torch.maximum(lower, torch.minimum(upper, adversarial)).detach()
    return adversarial


def attack_parameters() -> dict:
    return {
        "steps": STEPS,
        "alpha": ALPHA,
        "spectrum_samples": SPECTRUM_SAMPLES,
        "rho": RHO,
        "sigma": SIGMA,
        "gradient_mode": GRADIENT_MODE,
        "momentum_enabled": False,
        "di_enabled": False,
        "ti_enabled": False,
        "preserve_dc": False,
        "backward_passes_per_image": STEPS * SPECTRUM_SAMPLES,
        "source_weight_path": str(Path(MODEL_PATH).resolve()),
        "batch_size": BATCH_SIZE,
    }


def validate_parameters() -> None:
    if (
        type(STEPS) is not int
        or STEPS <= 0
        or type(SPECTRUM_SAMPLES) is not int
        or (SPECTRUM_SAMPLES <= 0)
    ):
        raise ValueError("STEPS and SPECTRUM_SAMPLES must be positive integers.")
    if not math.isfinite(EPS) or not 0 <= EPS <= 1:
        raise ValueError("EPS must be finite in [0,1].")
    if not math.isfinite(ALPHA) or not 0 <= ALPHA <= 1 or (EPS > 0 and ALPHA == 0):
        raise ValueError("ALPHA must be finite in [0,1] and positive when EPS is positive.")
    if not math.isfinite(RHO) or not 0 <= RHO <= 1:
        raise ValueError("RHO must be in [0,1].")
    if not math.isfinite(SIGMA) or SIGMA < 0:
        raise ValueError("SIGMA must be finite and nonnegative in [0,1] pixel coordinates.")
    if GRADIENT_MODE not in ("author", "paper"):
        raise ValueError("GRADIENT_MODE must be author or paper.")
    if (
        type(BATCH_SIZE) is not int
        or BATCH_SIZE <= 0
        or type(NUM_WORKERS) is not int
        or (NUM_WORKERS < 0)
    ):
        raise ValueError(
            "BATCH_SIZE must be a positive integer and NUM_WORKERS a nonnegative integer."
        )
    if IMAGE_SIZE != 224:
        raise ValueError("Inputs must be 224x224 without resizing; keep IMAGE_SIZE=224.")


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
    validate_parameters()
    device = torch.device(DEVICE)
    if device.type == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA is unavailable; check the PyTorch build or set DEVICE to cpu.")
    class_names = load_class_index()
    dataset = ImageFolderDataset()
    output_root = Path(OUTPUT_IMAGE_DIR).resolve()
    manifest_path = Path(MANIFEST_PATH).resolve()
    if output_root == dataset.root or dataset.root in output_root.parents:
        raise ValueError("The output directory must be outside the clean-image directory.")
    protected = {Path(MODEL_PATH).resolve(), Path(CLASS_INDEX_PATH).resolve()}
    protected.update((dataset.root / name for name in dataset.names))
    protected.update((output_root / Path(name).with_suffix(".png") for name in dataset.names))
    if manifest_path in protected:
        raise ValueError("MANIFEST_PATH must not overwrite an input file or a generated image.")
    if not OVERWRITE:
        if manifest_path.exists():
            raise FileExistsError(
                f"Manifest already exists: {manifest_path}; choose another output path."
            )
        for name in dataset.names:
            path = output_root / Path(name).with_suffix(".png")
            if path.exists():
                raise FileExistsError(
                    f"Generated image already exists: {path}; choose another output path."
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
    completed, successful = (0, 0)
    total_attack_seconds = 0.0
    print(f"{METHOD_NAME}:mode={GRADIENT_MODE},N={SPECTRUM_SAMPLES},rho={RHO},sigma={SIGMA:.8f}")
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
            "attack_success",
            "successful_images",
            "total_images",
            "ASR_percent",
        ]
        fields += list(attack_parameters()) + ["attack_seconds_per_image", "total_attack_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for clean, names in loader:
            clean = clean.to(device, non_blocking=True)
            with torch.no_grad():
                clean_predictions = model(normalize(clean)).argmax(dim=1)
            labels = clean_predictions.detach()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            adversarial = ssa(model, clean, labels)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            attack_seconds = time.perf_counter() - started
            total_attack_seconds += attack_seconds
            adv_u8 = quantize_images(clean, adversarial)
            with torch.no_grad():
                saved_pixels = adv_u8.to(device=device, dtype=clean.dtype) / 255
                adversarial_predictions = model(normalize(saved_pixels)).argmax(dim=1)
            cp, ap = (clean_predictions.cpu().tolist(), adversarial_predictions.cpu().tolist())
            for i, name in enumerate(names):
                generated = Path(name).with_suffix(".png").as_posix()
                save_png(adv_u8[i], output_root / generated)
                success = int(cp[i] != ap[i])
                completed += 1
                successful += success
                writer.writerow(
                    {
                        "row_type": "image",
                        "filename": name,
                        "generated_filename": generated,
                        "attack_label": cp[i],
                        "attack_class_name": class_names[cp[i]],
                        "label_source": "source_model_prediction",
                        "clean_prediction": cp[i],
                        "adversarial_prediction": ap[i],
                        "adversarial_class_name": class_names[ap[i]],
                        "method": METHOD_NAME,
                        "source_model": MODEL_NAME,
                        "eps": EPS,
                        "attack_success": success,
                        "successful_images": successful,
                        "total_images": completed,
                        "ASR_percent": f"{100 * successful / completed:.6f}",
                        **attack_parameters(),
                        "attack_seconds_per_image": attack_seconds / len(clean),
                        "total_attack_seconds": total_attack_seconds,
                    }
                )
            handle.flush()
            print(
                f"{METHOD_NAME}: processed {completed}/{len(dataset)}; cumulative source prediction-change rate {100 * successful / completed:.2f}%"
            )
        writer.writerow(
            {
                "row_type": "summary",
                "label_source": "source_model_prediction",
                "method": METHOD_NAME,
                "source_model": MODEL_NAME,
                "eps": EPS,
                "successful_images": successful,
                "total_images": completed,
                "ASR_percent": f"{100 * successful / completed:.6f}",
                **attack_parameters(),
                "attack_seconds_per_image": total_attack_seconds / completed,
                "total_attack_seconds": total_attack_seconds,
            }
        )
    print(f"Final source prediction-change rate: {100 * successful / completed:.2f}%")
    print(f"PNG directory: {output_root}\nSuccess manifest: {manifest_path}")


if __name__ == "__main__":
    main()
