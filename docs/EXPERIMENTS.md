# Experimental settings

## Shared protocol

Use naturally trained ResNet50 and DenseNet121 as sources, and one common-correct 1,000-image set across all tables and settings. Budgets are `4/255`, `8/255`, and `16/255`, with one run per configuration and no fixed random seed. The batch size is 16. Generation labels are fixed clean source predictions. Images and perturbation constraints use unnormalized RGB pixel coordinates in `[0,1]`. Model inference uses ImageNet channel normalization. Saved outputs are 8-bit RGB PNGs; rounding is constrained to the legal integer interval around the decoded clean image, preserving the pixel budget.

FGSM and TI-FGSM use one step of size `EPS`. The remaining methods use ten outer steps of size `EPS/10`. PGD has a uniform random start; the other methods initialize at the clean image. These are the evaluated implementation variants, adapted to the shared protocol, rather than claims to reproduce every original paper's complete experimental setup.

## Baseline settings

| Method | Settings |
|---|---|
| FGSM | One cross-entropy sign-gradient update; step `EPS`. |
| PGD | Ten steps; uniform initial pixel noise in `[-EPS, EPS]`, clipped to `[0,1]`; no momentum. |
| MI-FGSM | Momentum 1; per-image L1 gradient normalization. |
| DI-FGSM | No momentum; transform probability 0.5; nearest-neighbor resize to integer side 224 through 247, then zero-padding to 248; no resize back. The transform acts on normalized model inputs and is shared within the minibatch. |
| TI-FGSM | One step; no momentum; normalized 15-by-15 Gaussian kernel from 15 coordinates spanning `[-3,3]`; depthwise gradient smoothing with same-size zero padding before the sign update. |
| SI-NI-FGSM | Momentum 1; Nesterov lookahead; five intensity scales `1/(2**s)` for `s=0,...,4`, applied in pixel coordinates before normalization. |
| Admix | Three other-predicted-class donors per iteration; strength 0.2; five pixel-intensity scales; momentum 1. |
| VMI-FGSM | Momentum 1; 20 uniform neighbors in `[-1.5*EPS, 1.5*EPS]`; the variance estimate affects the following iteration. |
| NAA | Momentum 1; 30 integration points from a black baseline; attribution weight 1; `INTEGRATION_MODE="paper"`; feature layers `layer2` for ResNet50 and `features.denseblock2` for DenseNet121. |
| SSA | Basic spectrum-simulation iterative variant; 20 spectral samples, `RHO=0.5`, `SIGMA=EPS`; `GRADIENT_MODE="author"`; no momentum or additional DI/TI/Admix/intensity scaling. |
| MEF | Outer momentum 0.5; inner coefficient 0.9; 20 uniform neighbors with radius `2*EPS`; exploration coefficient 0.15; `MEF_IMPLEMENTATION="paper"`. |
| ResPA | Momentum 1; five uniform neighbors with radius `1.5*EPS`; EMA decay 0.4; fusion weight 0.4; residual radius `ALPHA`. |

For NAA, MEF, and SSA, branch names `paper` and `author` are script configuration identifiers. They do not imply that the repository files are official author releases or that different branches are interchangeable.

## MASS and ablations

Full MASS uses `STEPS=10`, `ALPHA=EPS/STEPS`, `MOMENTUM=1`, `ADMIX_COPIES=3`, `SCALE_COPIES=5`, `ADMIX_STRENGTH=0.2`, `RHO=0.5`, `SIGMA=EPS`, `SPECTRUM_SAMPLES=20`, and `GRADIENT_MODE="author"`. Sampling variants change only the spectral sample count to 1 or 5. Set every source, budget, and output path explicitly before each run.

The donor pool contains the same fixed clean inputs. Donors are uniformly sampled with replacement from entries whose clean source prediction differs from the attacked image's fixed class. The donor cache size is 128; cache contents do not restrict the sampling pool. Each donor is reused across its scales and spectral views within that iteration.

The DCT-II and its paired inverse act independently on each channel's spatial dimensions using the implementation's non-orthonormal convention. All mask coefficients, including DC, are randomized. Internal admixtures and reconstructions are not clipped. Reconstructed views are detached before intensity scaling; gradients retain the scale factor and normalization but do not pass through the spectral transformation. `GRADIENT_MODE="paper"` is a different, full-chain-rule experiment and should not be substituted silently.

The five-sample ablation reference uses the same averaging slots as the removals:

| File | Change relative to full N=5 |
|---|---|
| `MASS_full_N5.py` | No component removed. |
| `MASS_wo_Admix.py` | Admixture strength 0. |
| `MASS_wo_Momentum.py` | Momentum 0. |
| `MASS_wo_Scale.py` | All intensity-scale divisors set to 1. |
| `MASS_wo_Spectrum.py` | Gaussian noise and spectral-mask radius both 0. |

These are operational component removals, not a full factorial decomposition. `EXPECTED_IMAGE_COUNT=1000` is enabled in the ablation files. All five must be run with the chosen natural source and the same images/budget.

## Evaluation outputs

The summary CSV contains per-method/per-target `ASR`, `SuccessfulImages`, `Samples`, and prediction-agreement fields. The detail CSV contains paired filenames, clean and adversarial predictions, and a binary success flag. Report victim-specific counts even when all are 1,000. Average eight natural targets excluding the source, or four AT targets, with equal target weight.

The evaluator does not establish image-level ground-truth correctness. Its class-name JSON is a vocabulary, not image labels. The paper's selection condition makes prediction-change ASR equal to ground-truth ASR. Experiments on other inputs must retain the code's prediction-change interpretation unless correctness is independently established.
