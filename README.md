# ComfyUI Qwen3 VL AutoTagger

A ComfyUI custom node that generates Adobe Stock-style metadata (title + keywords) using `Qwen/Qwen3-VL-8B-Instruct`.
It saves tagged images with embedded XMP metadata via `exiftool` and returns them as ComfyUI outputs.

## Quick Start (Manual)

1. Copy this repository folder into `ComfyUI/custom_nodes/comfyui-qwen3-autotagger`.
2. Install Python requirements:

```bash
pip install -r custom_nodes/comfyui-qwen3-autotagger/requirements.txt
```

3. Install `exiftool` (required for XMP embedding) and ensure it is in `PATH`.
4. Restart ComfyUI.

## Quick Start (Git)

```bash
git clone https://github.com/ekkonwork/comfyui-qwen3-autotagger ComfyUI/custom_nodes/comfyui-qwen3-autotagger
pip install -r ComfyUI/custom_nodes/comfyui-qwen3-autotagger/requirements.txt
```

## Node

`Qwen3 VL AutoTagger`

The node ships with documentation (Node Docs). Open the Node Docs panel or hover inputs to see tooltips.

### Inputs

- `image` (IMAGE): ComfyUI image batch.
- `system_prompt` (STRING): Prompt that forces JSON output.
- `max_keywords` (INT): Keywords to keep (default 50).
- `max_new_tokens` (INT): Generation length.
- `temperature`, `top_p`, `repetition_penalty`: Sampling controls.
- `attempts` (INT): Retry count if JSON is invalid.
- `min_pixels`, `max_pixels` (INT): Vision resize constraints.
- `allow_resize` (BOOLEAN): Allow processor to resize images to valid patch sizes.
- `model_id` (STRING): Hugging Face model ID (default `Qwen/Qwen3-VL-8B-Instruct`).
- `auto_download` (BOOLEAN): Allow downloading the model on first run.
- `local_model` (CHOICE): Local model under `models/LLM` or `models/llm` (or `(manual)`).
- `local_model_path` (STRING): Full path to a local model folder (used when `local_model` is `(manual)`).
- `load_in_4bit` (BOOLEAN): Use 4-bit quantization if available.
- `write_xmp` (BOOLEAN): Save files and embed XMP metadata with `exiftool` (default: true).
- `require_exiftool` (BOOLEAN): Fail if `exiftool` is missing when `write_xmp` is enabled.
- `log_tags` (BOOLEAN): Print title + keyword preview in ComfyUI console.
- `output_dir` (STRING): Output directory (empty uses ComfyUI output dir).
- `output_format` (STRING): `jpg`, `png`, or `webp`.
- `file_prefix` (STRING): Output filename prefix.

### Outputs

- `image`: Original image batch (pass-through).
- `title`: Title per image (newline-separated for batch).
- `keywords`: Comma-separated keywords per image (newline-separated for batch).
- `json`: JSON per image (newline-separated for batch).
- `xmp_paths`: Saved file paths (newline-separated), empty if `write_xmp` is false.

## Output Behavior

When `write_xmp` is enabled, the node saves tagged images itself and returns them as ComfyUI outputs (no need to add `SaveImage`).
If you set a custom `output_dir` outside ComfyUI's output folder, the UI preview may not show the image.

## Model Download Size

The default model (`Qwen/Qwen3-VL-8B-Instruct`) downloads about 17.5 GB of weights in total (roughly 16.3 GiB).

## Performance

On a Colab T4, a single image typically takes about 60 seconds to auto-tag (varies with resolution and settings).

## Local Model Loading

If you already have the model downloaded, you can place it under:

- `ComfyUI/models/LLM/<YourModelFolder>` or `ComfyUI/models/llm/<YourModelFolder>`

Note: `LLM` is not part of the official default model subfolders in ComfyUI, so this is a convenience convention for this node.
You can instead use a custom location via `local_model_path`.

Then set:

- `auto_download = false`
- `local_model = LLM/<YourModelFolder>` (or use `local_model_path` for a custom location)

## Example Workflows

- `example_workflows/Qwen3VLAutoTagger_minimal.json`
- `example_workflows/Qwen3VLAutoTagger_api.json`

## Publishing Notes

This repository is structured as a ComfyUI custom node root. If you see a nested folder named
`comfyui-qwen3-autotagger`, it is a legacy artifact from early development and can be removed later.

## License

MIT. See `LICENSE`.
