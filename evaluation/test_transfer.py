# test_transfer
# Fill every "......" path and review the configuration before running.
# See README.md and docs/EXPERIMENTS.md for the evaluated protocol.

from pathlib import Path
import csv
import json
import math
import re
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ORIGINAL_IMAGE_DIR = "......"
METHOD_IMAGE_DIRS = {
    "FGSM": "......",
    "PGD": "......",
    "MI-FGSM": "......",
    "DI-FGSM": "......",
    "TI-FGSM": "......",
    "SI-NI-FGSM": "......",
    "Admix": "......",
    "VMI-FGSM": "......",
    "NAA": "......",
    "SSA": "......",
    "MEF": "......",
    "ResPA": "......",
    "Admix_SSA_N20": "......",
    "Admix_SSA_N1": "......",
    "Admix_SSA_N5": "......",
    "MASS_wo_Admix": "......",
    "MASS_wo_Momentum": "......",
    "MASS_wo_Scale": "......",
    "MASS_wo_Spectrum": "......",
}
CLASS_INDEX_PATH = "......"
RESULT_CSV_PATH = "......"
DETAIL_CSV_PATH = "......"
DEVICE = "cuda:0"
BATCH_SIZE = 16
NUM_WORKERS = 4
PIN_MEMORY = True
OVERWRITE = False
NUM_CLASSES = 1000
IMAGE_SIZE = 224
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_SPECS = [
    {
        "display_name": "VGG11",
        "backend": "torchvision",
        "architecture": "vgg11",
        "weight_path": "......",
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
    {
        "display_name": "GoogLeNet",
        "backend": "torchvision",
        "architecture": "googlenet",
        "weight_path": "......",
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "transform_input": True,
    },
    {
        "display_name": "ResNet18",
        "backend": "torchvision",
        "architecture": "resnet18",
        "weight_path": "......",
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
    {
        "display_name": "DenseNet121",
        "backend": "torchvision",
        "architecture": "densenet121",
        "weight_path": "......",
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
    {
        "display_name": "MobileNetV2",
        "backend": "torchvision",
        "architecture": "mobilenet_v2",
        "weight_path": "......",
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
    {
        "display_name": "ViT-B/16",
        "backend": "torchvision",
        "architecture": "vit_b_16",
        "weight_path": "......",
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
    {
        "display_name": "Swin-T",
        "backend": "torchvision",
        "architecture": "swin_t",
        "weight_path": "......",
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
    {
        "display_name": "ConvNeXt-Tiny",
        "backend": "torchvision",
        "architecture": "convnext_tiny",
        "weight_path": "......",
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
]
COMPUTE_IMAGE_METRICS = True
SSIM_WINDOW_SIZE = 11
SSIM_SIGMA = 1.5
SSIM_K1 = 0.01
SSIM_K2 = 0.03


def scan_images(root: Path) -> dict[str, tuple[Path, str]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    indexed: dict[str, tuple[Path, str]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(root)
        key = relative.with_suffix("").as_posix().casefold()
        if key in indexed:
            raise ValueError(
                f"Ambiguous paired filenames: {indexed[key][0]} and {path}; use distinct filenames."
            )
        indexed[key] = (path, relative.as_posix())
    if not indexed:
        raise ValueError(f"No readable images in directory: {root}")
    return indexed


class PairedImageDataset(Dataset):

    def __init__(self, originals: dict[str, tuple[Path, str]], generated_root: Path):
        generated = scan_images(generated_root)
        missing = sorted(set(originals) - set(generated))
        extra = sorted(set(generated) - set(originals))
        if missing or extra:
            raise ValueError(
                f"Clean and adversarial image sets do not match: {generated_root}\nMissing {len(missing)} images, for example {missing[:5]}; extra {len(extra)} images, for example {extra[:5]}."
            )
        self.pairs = [(originals[key], generated[key]) for key in sorted(originals)]
        self.transform = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.pairs)

    def read_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(
                    f"Image size must be {IMAGE_SIZE}x{IMAGE_SIZE} ; actual {image.size}:{path}"
                )
            return self.transform(image)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        (clean_path, clean_name), (adv_path, adv_name) = self.pairs[index]
        return (self.read_image(clean_path), self.read_image(adv_path), clean_name, adv_name)


def make_loader(dataset: PairedImageDataset, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and device.type == "cuda",
    )


def load_class_index() -> dict[int, str]:
    if CLASS_INDEX_PATH is None:
        return {}
    with Path(CLASS_INDEX_PATH).open(encoding="utf-8-sig") as handle:
        entries = json.load(handle)
    if not isinstance(entries, dict) or set(entries) != {str(i) for i in range(NUM_CLASSES)}:
        raise ValueError("The class JSON must contain every key from 0 through NUM_CLASSES-1.")
    result: dict[int, str] = {}
    for key, value in entries.items():
        if isinstance(value, str) and value.strip():
            result[int(key)] = value
        elif isinstance(value, list) and len(value) == 2 and isinstance(value[1], str):
            result[int(key)] = value[1]
        else:
            raise ValueError(f"Invalid class JSON entry: {key}: {value}")
    return result


def extract_state_dict(checkpoint: object) -> dict:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a state_dict or a dictionary containing one.")
    state = checkpoint
    for key in ("state_dict", "model_state_dict", "model", "state_dict_ema", "model_ema"):
        if isinstance(checkpoint.get(key), dict):
            state = checkpoint[key]
            break
    if not state or not all(
        (isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items())
    ):
        raise TypeError(
            "The selected state dictionary must contain only string keys and Tensor values; export model.state_dict()."
        )
    while True:
        prefix = next(
            (p for p in ("module.", "_orig_mod.") if all((k.startswith(p) for k in state))), None
        )
        if prefix is None:
            break
        state = {key[len(prefix) :]: value for key, value in state.items()}
    return state


def load_model(spec: dict, device: torch.device) -> nn.Module:
    path = Path(spec["weight_path"])
    if not path.is_file():
        raise FileNotFoundError(f"{spec['display_name']} local checkpoint does not exist: {path}")
    state = extract_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    architecture = spec["architecture"]
    if spec["backend"] == "torchvision":
        if architecture == "densenet121":
            pattern = re.compile(
                "^(.*denselayer\\d+\\.(?:norm|relu|conv))\\.([12])\\.(weight|bias|running_mean|running_var|num_batches_tracked)$"
            )
            for key in list(state):
                match = pattern.match(key)
                if match:
                    updated = match.group(1) + match.group(2) + "." + match.group(3)
                    if updated in state:
                        raise ValueError(
                            f"Duplicate legacy/current DenseNet keys: {key} and {updated}"
                        )
                    state[updated] = state.pop(key)
        constructors = {
            "vgg11": models.vgg11,
            "resnet50": models.resnet50,
            "resnet18": models.resnet18,
            "densenet121": models.densenet121,
            "mobilenet_v2": models.mobilenet_v2,
            "vit_b_16": models.vit_b_16,
            "swin_t": models.swin_t,
            "convnext_tiny": models.convnext_tiny,
        }
        if architecture == "googlenet":
            has_aux = any((k.startswith(("aux1.", "aux2.")) for k in state))
            model = models.googlenet(
                weights=None,
                num_classes=NUM_CLASSES,
                aux_logits=has_aux,
                init_weights=False,
                transform_input=spec.get("transform_input", True),
            )
        elif architecture in constructors:
            kwargs = {"image_size": IMAGE_SIZE} if architecture == "vit_b_16" else {}
            model = constructors[architecture](weights=None, num_classes=NUM_CLASSES, **kwargs)
        else:
            raise ValueError(f"Unknown torchvision architecture: {architecture}")
    elif spec["backend"] == "timm":
        import timm

        model = timm.create_model(architecture, pretrained=False, num_classes=NUM_CLASSES)
    else:
        raise ValueError("backend must be torchvision or timm.")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"{spec['display_name']}Checkpoint mismatch; check backend, architecture, and NUM_CLASSES.\n{error}"
        ) from error
    return model.to(device).eval().requires_grad_(False)


def normalize(pixels: torch.Tensor, spec: dict) -> torch.Tensor:
    mean = pixels.new_tensor(spec["mean"]).view(1, 3, 1, 1)
    std = pixels.new_tensor(spec["std"]).view(1, 3, 1, 1)
    return (pixels - mean) / std


def predict(model: nn.Module, pixels: torch.Tensor, spec: dict) -> torch.Tensor:
    logits = model(normalize(pixels, spec))
    if not isinstance(logits, torch.Tensor) or logits.shape != (len(pixels), NUM_CLASSES):
        raise ValueError(
            f"{spec['display_name']} output shape must be [batch, {NUM_CLASSES}] logits."
        )
    if not torch.isfinite(logits).all():
        raise FloatingPointError(f"{spec['display_name']} output contains NaN/Inf.")
    return logits.argmax(dim=1)


def image_metrics(
    clean: torch.Tensor, adv: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    clean, adv = (clean.to(torch.float64), adv.to(torch.float64))
    difference = adv - clean
    l2 = difference.flatten(1).norm(p=2, dim=1)
    mse = difference.square().mean((1, 2, 3))
    psnr = torch.where(mse == 0, torch.full_like(mse, float("inf")), -10 * torch.log10(mse))
    coordinates = torch.arange(SSIM_WINDOW_SIZE, device=clean.device, dtype=clean.dtype)
    coordinates -= SSIM_WINDOW_SIZE // 2
    gaussian = torch.exp(-coordinates.square() / (2 * SSIM_SIGMA**2))
    gaussian /= gaussian.sum()
    kernel = (gaussian[:, None] * gaussian[None, :]).expand(3, 1, -1, -1).contiguous()
    moments = F.conv2d(
        torch.cat((clean, adv, clean.square(), adv.square(), clean * adv)), kernel, groups=3
    )
    mx, my, ex2, ey2, exy = moments.chunk(5)
    vx, vy, covariance = (ex2 - mx.square(), ey2 - my.square(), exy - mx * my)
    c1, c2 = (SSIM_K1**2, SSIM_K2**2)
    ssim_map = (
        (2 * mx * my + c1)
        * (2 * covariance + c2)
        / ((mx.square() + my.square() + c1) * (vx + vy + c2))
    )
    return (l2, ssim_map.mean((1, 2, 3)), psnr)


@torch.inference_mode()
def summarize_image_metrics(dataset: PairedImageDataset, device: torch.device) -> dict:
    if not COMPUTE_IMAGE_METRICS:
        return {"L2": "", "SSIM": "", "PSNR": ""}
    totals = [0.0, 0.0, 0.0]
    for clean, adv, _, _ in make_loader(dataset, device):
        values = image_metrics(clean.to(device), adv.to(device))
        for i, values_per_image in enumerate(values):
            totals[i] += values_per_image.sum().item()
    return {name: value / len(dataset) for name, value in zip(("L2", "SSIM", "PSNR"), totals)}


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    spec: dict,
    method: str,
    dataset: PairedImageDataset,
    device: torch.device,
    class_names: dict[int, str],
    details: csv.DictWriter,
) -> dict:
    completed, successful = (0, 0)
    for clean, adv, clean_names, adv_names in make_loader(dataset, device):
        clean_predictions = predict(model, clean.to(device, non_blocking=True), spec).cpu().tolist()
        adv_predictions = predict(model, adv.to(device, non_blocking=True), spec).cpu().tolist()
        for clean_name, adv_name, cp, ap in zip(
            clean_names, adv_names, clean_predictions, adv_predictions
        ):
            success = int(cp != ap)
            completed += 1
            successful += success
            details.writerow(
                {
                    "Method": method,
                    "Model": spec["display_name"],
                    "filename": clean_name,
                    "generated_filename": adv_name,
                    "clean_prediction": cp,
                    "adversarial_prediction": ap,
                    "clean_class_name": class_names.get(cp, ""),
                    "adversarial_class_name": class_names.get(ap, ""),
                    "attack_success": success,
                }
            )
        print(
            f"{spec['display_name']} / {method}:{completed}/{len(dataset)}; successful {successful}; prediction-change rate {100 * successful / completed:.2f}%"
        )
    accuracy = 100 * (completed - successful) / completed
    return {
        "Method": method,
        "Model": spec["display_name"],
        "Accuracy": accuracy,
        "ASR": 100 * successful / completed,
        "SuccessfulImages": successful,
        "Samples": completed,
        "PredictionAgreementPercent": accuracy,
        "Accuracy_definition": "agreement_with_target_clean_prediction",
        "Accuracy_unit": "percent",
        "ASR_definition": "target_prediction_change",
        "ASR_unit": "percent",
        "Backend": spec["backend"],
        "Architecture": spec["architecture"],
        "WeightPath": str(Path(spec["weight_path"]).resolve()),
        "Mean": json.dumps(spec["mean"]),
        "Std": json.dumps(spec["std"]),
        "GoogLeNetTransformInput": (
            spec.get("transform_input", True)
            if spec["backend"] == "torchvision" and spec["architecture"] == "googlenet"
            else ""
        ),
    }


def validate_parameters() -> None:
    if (
        not isinstance(BATCH_SIZE, int)
        or BATCH_SIZE <= 0
        or (not isinstance(NUM_WORKERS, int))
        or (NUM_WORKERS < 0)
    ):
        raise ValueError(
            "BATCH_SIZE must be a positive integer and NUM_WORKERS a nonnegative integer."
        )
    if IMAGE_SIZE != 224:
        raise ValueError("Evaluation uses 224x224 images directly; keep IMAGE_SIZE=224.")
    if not MODEL_SPECS or not METHOD_IMAGE_DIRS:
        raise ValueError("MODEL_SPECS and METHOD_IMAGE_DIRS must not be empty.")
    if len({spec["display_name"] for spec in MODEL_SPECS}) != len(MODEL_SPECS):
        raise ValueError("Model display names must be unique.")
    for spec in MODEL_SPECS:
        if spec["backend"] not in ("torchvision", "timm"):
            raise ValueError(f"Invalid model backend: {spec}")
        if len(spec["mean"]) != 3 or len(spec["std"]) != 3:
            raise ValueError("Each model must have three RGB mean and standard-deviation values.")
        if not all((math.isfinite(v) for v in spec["mean"])) or not all(
            (math.isfinite(v) and v > 0 for v in spec["std"])
        ):
            raise ValueError("mean must be finite; std must be finite and positive.")
        if not Path(spec["weight_path"]).is_file():
            raise FileNotFoundError(
                f"{spec['display_name']}Checkpoint does not exist: {spec['weight_path']}"
            )
    if COMPUTE_IMAGE_METRICS:
        if (
            not isinstance(SSIM_WINDOW_SIZE, int)
            or SSIM_WINDOW_SIZE < 1
            or SSIM_WINDOW_SIZE % 2 == 0
            or (SSIM_WINDOW_SIZE > IMAGE_SIZE)
        ):
            raise ValueError(
                "SSIM_WINDOW_SIZE must be a positive odd integer no larger than the image size."
            )
        if not all((math.isfinite(v) and v > 0 for v in (SSIM_SIGMA, SSIM_K1, SSIM_K2))):
            raise ValueError("SSIM_SIGMA, SSIM_K1, and SSIM_K2 must be finite and positive.")


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
        raise RuntimeError(
            "CUDA is unavailable; install a CUDA build of PyTorch or set DEVICE to cpu."
        )
    class_names = load_class_index()
    originals = scan_images(Path(ORIGINAL_IMAGE_DIR).resolve())
    datasets = {
        method: PairedImageDataset(originals, Path(path).resolve())
        for method, path in METHOD_IMAGE_DIRS.items()
    }
    result_path, detail_path = (Path(RESULT_CSV_PATH).resolve(), Path(DETAIL_CSV_PATH).resolve())
    if result_path == detail_path:
        raise ValueError("Summary and per-image CSV paths must be different.")
    input_files = {Path(spec["weight_path"]).resolve() for spec in MODEL_SPECS}
    if CLASS_INDEX_PATH is not None:
        input_files.add(Path(CLASS_INDEX_PATH).resolve())
    for dataset in datasets.values():
        for (clean_path, _), (adv_path, _) in dataset.pairs:
            input_files.update((clean_path.resolve(), adv_path.resolve()))
    for path in (result_path, detail_path):
        if path in input_files:
            raise ValueError(f"A CSV path must not overwrite an input file: {path}")
        if path.exists() and (not OVERWRITE):
            raise FileExistsError(
                f"Result already exists: {path}; choose another output path or set OVERWRITE=True."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
    print(
        "Each target uses its own clean predictions as references. Accuracy is prediction agreement; ASR is prediction change. Both are percentages."
    )
    quality = {
        method: summarize_image_metrics(dataset, device) for method, dataset in datasets.items()
    }
    result_fields = [
        "Method",
        "Model",
        "Accuracy",
        "ASR",
        "L2",
        "SSIM",
        "PSNR",
        "SuccessfulImages",
        "Samples",
        "PredictionAgreementPercent",
        "Accuracy_definition",
        "Accuracy_unit",
        "ASR_definition",
        "ASR_unit",
        "Backend",
        "Architecture",
        "WeightPath",
        "Mean",
        "Std",
        "GoogLeNetTransformInput",
    ]
    detail_fields = [
        "Method",
        "Model",
        "filename",
        "generated_filename",
        "clean_prediction",
        "adversarial_prediction",
        "clean_class_name",
        "adversarial_class_name",
        "attack_success",
    ]
    with result_path.open("w", encoding="utf-8-sig", newline="") as result_handle, detail_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as detail_handle:
        results = csv.DictWriter(result_handle, fieldnames=result_fields)
        details = csv.DictWriter(detail_handle, fieldnames=detail_fields)
        results.writeheader()
        details.writeheader()
        for spec in MODEL_SPECS:
            print(f"Loading local model: {spec['display_name']}; checkpoint: {spec['weight_path']}")
            model = load_model(spec, device)
            try:
                for method, dataset in datasets.items():
                    row = evaluate(model, spec, method, dataset, device, class_names, details)
                    row.update(quality[method])
                    results.writerow(row)
                    result_handle.flush()
                    detail_handle.flush()
            finally:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    print(f"Evaluation complete.\nSummary: {result_path}\nPer-image results: {detail_path}")


if __name__ == "__main__":
    main()
