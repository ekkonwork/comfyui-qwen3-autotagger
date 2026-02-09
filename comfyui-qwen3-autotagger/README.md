# ComfyUI Qwen3 VL AutoTagger

A ComfyUI custom node that generates Adobe Stock-style metadata (title + keywords) using `Qwen/Qwen3-VL-8B-Instruct`. It can optionally save images with embedded XMP metadata via `exiftool`.

## Installation

1. Copy this folder into `ComfyUI/custom_nodes/comfyui-qwen3-autotagger`.
2. Install Python requirements:

```bash
pip install -r custom_nodes/comfyui-qwen3-autotagger/requirements.txt
```

Optional (for XMP embedding): install `exiftool` and make sure it is available in your system `PATH`.
3. Restart ComfyUI.

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
- `output_dir` (STRING): Output directory (empty uses ComfyUI output dir).
- `output_format` (STRING): `jpg`, `png`, or `webp`.
- `file_prefix` (STRING): Output filename prefix.

### Outputs

- `image`: Original image batch (pass-through).
- `title`: Title per image (newline-separated for batch).
- `keywords`: Comma-separated keywords per image (newline-separated for batch).
- `json`: JSON per image (newline-separated for batch).
- `xmp_paths`: Saved file paths (newline-separated), empty if `write_xmp` is false.

## Notes

- This model is large; use a GPU if possible.
- If `bitsandbytes` is unavailable, the node will load without 4-bit quantization.
- 4-bit quantization requires CUDA; on CPU it will fall back to full precision.
- If `exiftool` is missing, images will be saved without XMP metadata.

## Local Model Loading

If you already have the model downloaded, you can place it under:

- `ComfyUI/models/LLM/<YourModelFolder>` or `ComfyUI/models/llm/<YourModelFolder>`

Note: `LLM` is not part of the official default model subfolders in ComfyUI, so this is a convenience convention for this node. You can instead use a custom location via `local_model_path`.

Then set:

- `auto_download = false`
- `local_model = LLM/<YourModelFolder>` (or use `local_model_path` for a custom location)

## Example Workflow

An example workflow is included in `example_workflows/`.

For headless/API runs, use `example_workflows/Qwen3VLAutoTagger_api.json`.

## Output Behavior

When `write_xmp` is enabled, the node saves tagged images itself and returns them as ComfyUI outputs (no need to add `SaveImage`).
If you set a custom `output_dir` outside ComfyUI's output folder, the UI preview may not show the image.
Open it in ComfyUI via `Workflow -> Browse Workflow Templates`.
