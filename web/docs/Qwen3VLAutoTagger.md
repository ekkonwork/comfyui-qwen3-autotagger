# Qwen3 VL AutoTagger

Generates Adobe Stock-style metadata (title + keywords) from an input image using Qwen3-VL.

## Inputs
- `image`: Input image batch.
- `system_prompt`: System prompt that enforces JSON output.
- `max_keywords`: Keep up to N keywords.
- `max_new_tokens`: Generation length.
- `temperature`: Sampling temperature (0 disables sampling).
- `top_p`: Nucleus sampling.
- `repetition_penalty`: Discourage repeats.
- `attempts`: Retry count if JSON is invalid.
- `min_pixels`: Minimum vision resolution budget.
- `max_pixels`: Maximum vision resolution budget.
- `allow_resize`: Allow processor to resize images to valid patch sizes.
- `model_id`: HF model id (used when auto-download is enabled).
- `auto_download`: Allow downloading the model from Hugging Face on first run.
- `local_model`: Select a local model folder from `models/LLM` (or `models/llm`).
- `local_model_path`: Full path to a local model folder (used when `local_model` is `(manual)`).
- `load_in_4bit`: Enable 4-bit quantization (CUDA only).
- `write_xmp`: Embed XMP metadata via `exiftool` (default: true).
- `require_exiftool`: Fail if `exiftool` is missing when `write_xmp` is enabled.
- `log_tags`: Print title + keyword preview in ComfyUI console.
- `output_dir`: Save directory. Empty uses ComfyUI output dir.
- `output_format`: Output format (`jpg`, `png`, `webp`).
- `file_prefix`: Output filename prefix.

## Outputs
- `image`: Original image batch (pass-through).
- `title`: One title per line.
- `keywords`: Comma-separated keywords per line.
- `json`: JSON per line.
- `xmp_paths`: Saved file paths per line (empty if XMP is off).

## Notes
- 4-bit quantization requires CUDA.
- When `write_xmp` is enabled, the node saves tagged images itself and returns them as UI outputs (no separate SaveImage needed).
- When `write_xmp` is disabled, the node does not save files.
- Saved filenames are auto-incremented and existing files are not overwritten.
- Default model download size is ~17.5 GB; on a Colab T4, a single image is usually ~60s.
- If `exiftool` is missing, images are saved without XMP metadata.
- If `output_dir` is outside ComfyUI output, UI preview may not show saved images.
- You can run `python install.py` on Linux to auto-install exiftool (uses apt-get).
