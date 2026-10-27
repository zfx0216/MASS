# Public repository preparation

This distribution contains 21 experiment scripts: all 20 supplied attack/ablation files and one configurable evaluator. The original source files were not modified. The evaluator replaces the four equivalent evaluation presets; select target models and image directories through its configuration.

Changes made for this distribution:

- Use English directory names: `attacks`, `ablations`, and `evaluation`.
- Remove the original comments/docstrings and translate Chinese runtime messages into English. Retain available method reference links in script headers and document implementation details centrally.
- Replace all absolute machine-specific paths and input-path prompts with the literal `......`, including paths nested in evaluation configurations.
- Add an early check that reports unresolved path fields before accessing models or generating output.
- Change the four component-removal scripts' stale `EXPECTED_IMAGE_COUNT=300` defaults to 1,000, matching the author's confirmed paper protocol.
- Change the full N=5 reference script's default `MODEL_NAME` from ResNet18 to ResNet50. It still requires the user to supply the matching checkpoint path. The paper sources remain naturally trained ResNet50 and DenseNet121.
- Preserve the original other numerical settings and the attack/evaluation computations. Retain one shared evaluation implementation; configuration examples for natural and AT targets are in README.md.
- Expand the common evaluator's method-directory list to include all 19 main, sampling, and ablation labels. Remove unused entries when running a smaller evaluation. Format the public Python files for readability.

`CODE_INVENTORY.json` records script-level source and public-file SHA-256 hashes and the two types of configuration adjustment. The packaging process compared normalized Python syntax trees after removing the new path guard, so functional changes were limited to path placeholders, translated strings, removed docstrings, and the documented configuration defaults. The scripts also pass syntax compilation and repository checks. These checks are not a re-run of the attacks or validation of the manuscript's numerical results.

The software versions record the author's supplied experimental environment. No model weights, images, fabricated results, or unrelated environment packages are included. Choose a repository license only after confirming ownership and any applicable upstream terms; this package does not assign a new license to third-party work.
