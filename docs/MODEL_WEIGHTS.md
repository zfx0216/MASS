# Model configuration and local checkpoints

Weights are not bundled. Obtain checkpoints from their original providers under the applicable terms and set each local `MODEL_PATH` or `weight_path` to its location. Every machine-specific path in the supplied Python files is intentionally `......`.

## Natural models

| Display name | torchvision architecture | Local filename recorded in the supplied code |
|---|---|---|
| VGG11 | `vgg11` | `vgg11-8a719046.pth` |
| GoogLeNet | `googlenet` | `googlenet-1378be20.pth` |
| ResNet18 | `resnet18` | `resnet18-offical.pth` |
| ResNet50 | `resnet50` | `resnet50-0676ba61.pth` |
| DenseNet121 | `densenet121` | `densenet121_true.pth` |
| MobileNetV2 | `mobilenet_v2` | `mobilenet_v2-b0353104.pth` |
| ViT-B/16 | `vit_b_16` | `vit_b_16-c867db91.pth` |
| Swin-T | `swin_t` | `swin_t-704ceda3.pth` |
| ConvNeXt-Tiny | `convnext_tiny` | `convnext_tiny-983f1562.pth` |

The custom local names `resnet18-offical.pth` and `densenet121_true.pth` do not independently establish an upstream release identifier. Do not replace them with guessed official weight versions. The inspected source checkpoint hashes are:

```text
resnet50-0676ba61.pth
0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a

densenet121_true.pth
d30c15e1fa83e941a44f2e63c41ec2a11fd4715f5c0e8b792115d00a2286039d
```

These hashes identify the inspected local files; they are not a provenance claim about every historical run. Model state dictionaries are loaded strictly. The evaluator handles legacy DenseNet key naming. GoogLeNet uses the recorded `transform_input=True` setting; retain that configuration when using the corresponding checkpoint. Attack constructors and evaluation constructors support different optional model lists; the manuscript sources are ResNet50 and DenseNet121.

## AT targets

| Paper display name | Local filename | Architecture |
|---|---|---|
| Fast-AT-R50 | `Wong2020Fast.pt` | `resnet50` |
| Eng-AT-R50 | `Engstrom2019Robustness.pt` | `resnet50` |
| Salman-AT-R18 | `Salman2020Do_R18.pt` | `resnet18` |
| Salman-AT-R50 | `Salman2020Do_R50.pt` | `resnet50` |

The evaluator's historical CSV names `AT-R50-Fast` and `AT-R50-Eng` correspond to Fast-AT-R50 and Eng-AT-R50. These checkpoints are targets only. The included evaluator loads them as state dictionaries into ordinary torchvision networks, with external ImageNet normalization and direct 224-by-224 inputs. It does not instantiate RobustBench normalization wrappers or apply their model-card spatial transforms.

Model-card preprocessing differs: Fast-AT-R50 uses Crop288, while the other three use Res256Crop224. Those transforms are not the implemented protocol here. If evaluating with RobustBench's loader instead, inspect its internal normalization and follow its prescribed geometry; do not add duplicate normalization or reuse the adapted-protocol results as though they came from that pipeline.
