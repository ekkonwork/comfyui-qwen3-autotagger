# ComfyUI Qwen3 VL AutoTagger

[![Donate on Boosty](https://img.shields.io/badge/Donate-Boosty-F15F2C?style=for-the-badge)](https://boosty.to/ekkonwork/donate)
[![Hire Me on LinkedIn](https://img.shields.io/badge/Hire%20Me-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mikhail-kuznetsov-14304433b)

![Qwen3 VL AutoTagger](assets/banner_tagger_for_stock.webp)

Generate Adobe Stock-style title + keywords with Qwen3-VL and embed XMP metadata directly into outputs.

## Standalone CLI Version

Need the same tagging pipeline without ComfyUI? Use the standalone CLI project:

[![CLI Repository](https://img.shields.io/badge/Standalone-CLI-1F6FEB?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ekkonwork/qwen3-vl-autotagger-cli)
[![Run CLI In Colab](https://img.shields.io/badge/Run%20CLI-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=black)](https://colab.research.google.com/github/ekkonwork/qwen3-vl-autotagger-cli/blob/main/Colab_T4_CLI_Prod.ipynb)

## Highlights

- Auto-tags images (title + ~60 keywords)
- Saves tagged images with XMP metadata when `write_xmp` is enabled (no `SaveImage` needed)
- Headless/API workflow included
- Auto-downloads `Qwen/Qwen3-VL-8B-Instruct` on first run when `auto_download=true` (default)
- Optional 4-bit quantization (CUDA)

## Screenshots

Node UI
![Node UI](assets/Node_only_screen.png)

Workflow Example
![Workflow Example](assets/Workflow_Screen_new.png)

Embedded XMP Metadata
![Metadata Screenshot](assets/metadata_screenshot.png)

Adobe Stock Result Example 1
![Adobe Stock Result 1](assets/on_stock_tags1.png)

Adobe Stock Result Example 2
![Adobe Stock Result 2](assets/on_stock_tags2.png)

## Quick Start (Manual)

1. Copy this repository folder into `ComfyUI/custom_nodes/comfyui-qwen3-autotagger`.
2. Install Python requirements:

```bash
pip install -r custom_nodes/comfyui-qwen3-autotagger/requirements.txt
```

3. Install `exiftool` (required for XMP embedding) and ensure it is in `PATH`.
   - On Linux you can also run `python install.py` to auto-install (uses `apt-get`).
4. Restart ComfyUI.

## Quick Start (Git)

```bash
git clone https://github.com/ekkonwork/comfyui-qwen3-autotagger ComfyUI/custom_nodes/comfyui-qwen3-autotagger
pip install -r ComfyUI/custom_nodes/comfyui-qwen3-autotagger/requirements.txt
python ComfyUI/custom_nodes/comfyui-qwen3-autotagger/install.py
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
- `min_pixels`, `max_pixels` (INT): Vision resize constraints by total pixel budget (area), not by max side length.
- Important: `max_pixels` is the total area `W*H`, so for about `1024` px on the long side (near-square) use `max_pixels=1024*1024` (`1048576`), not `1024`.
- Current defaults: `min_pixels=200704`, `max_pixels=571536` (about `756x756` area for square images).
- `allow_resize` (BOOLEAN): Allow processor to resize images to valid patch sizes (recommended to prevent OOM on very large inputs).
- `model_id` (STRING): Hugging Face model ID (default `Qwen/Qwen3-VL-8B-Instruct`).
- `auto_download` (BOOLEAN): Allow downloading the model on first run.
- `local_model` (CHOICE): Select a local model folder from `models/LLM` or `models/llm` (or `(manual)`).
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
When `write_xmp` is disabled, the node does not save files.
Saved filenames are auto-incremented (`file_prefix_00000`, `file_prefix_00001`, ...) and existing files are not overwritten.
If you set a custom `output_dir` outside ComfyUI's output folder, the UI preview may not show the image.

## Model Download Size

By default (`auto_download=true`), the node automatically downloads `Qwen/Qwen3-VL-8B-Instruct` on first run.
The download size is about 17.5 GB of weights in total (roughly 16.3 GiB).

## Performance

On a Colab T4, a single image typically takes about 60 seconds to auto-tag (varies with resolution and settings).

## Local Model Selection in Node

If you already have the model downloaded, place it under:

- `ComfyUI/models/LLM/<YourModelFolder>` or `ComfyUI/models/llm/<YourModelFolder>`

Then in the node set:

- `auto_download = false`
- `local_model = LLM/<YourModelFolder>` or `llm/<YourModelFolder>` (exactly as shown in the node dropdown)
- or use `local_model_path` for a custom location

## Support

If this node saves you time, you can support development on Boosty:

[![Donate](assets/Donate_Banner.webp)](https://boosty.to/ekkonwork/donate)

[![Donate on Boosty](https://img.shields.io/badge/Donate-Boosty-F15F2C?style=for-the-badge)](https://boosty.to/ekkonwork/donate)

- Boosty (donate): `https://boosty.to/ekkonwork/donate`

### Crypto Donations (Telegram Wallet)

[![Donate via Telegram Wallet](https://img.shields.io/badge/Donate-Telegram%20Wallet-2AABEE?style=for-the-badge&logo=telegram&logoColor=white)](#crypto-donations-telegram-wallet)
[![Open Telegram Wallet](https://img.shields.io/badge/Open-Wallet-26A17B?style=for-the-badge)](https://t.me/wallet)

![Telegram Wallet QR](assets/telegram_wallet_qr.png)

Scan this QR code in your wallet app to copy the donation address:

- TON: `UQAMPvqduXVWyax325-zqk81rTwNG1bRhCvXPyIs7eeIxEVp`
- USDT (TON): `UQAMPvqduXVWyax325-zqk81rTwNG1bRhCvXPyIs7eeIxEVp`
- Memo/Tag: check the Wallet receive screen before sending.

## Hire Me

[![Hire Me](assets/Hire_Me_banner.webp)](https://www.linkedin.com/in/mikhail-kuznetsov-14304433b)

[![Hire Me on LinkedIn](https://img.shields.io/badge/Hire%20Me-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mikhail-kuznetsov-14304433b)

- English: `B2` (text-first communication).
- Hiring (full-time/long-term): prefer written communication; for live calls, Russian-speaking teams are preferred.
- Project work: open to worldwide async collaboration.
- Email: `ekkonwork@gmail.com`
- Telegram: `@Mikhail_ML_ComfyUI`
- LinkedIn: `https://www.linkedin.com/in/mikhail-kuznetsov-14304433b`
- Boosty: `https://boosty.to/ekkonwork/donate`

## Example Workflows

- `example_workflows/Qwen3VLAutoTagger_minimal.json`
- `example_workflows/Qwen3VLAutoTagger_api.json`

## Troubleshooting (Environment)

- `exiftool` may be missing or not available in `PATH`.
- CUDA/driver/VRAM setup can differ between machines.
- `bitsandbytes` may fail to install or initialize on some systems. The node now retries model loading without 4-bit automatically.
- If you see `Qwen3VLForConditionalGeneration is not available`, reinstall dependencies from `requirements.txt` (latest `transformers` is required for Qwen3-VL).
- If you see model load errors, run:
  - `pip uninstall -y transformers qwen-vl-utils`
  - `pip install -U git+https://github.com/huggingface/transformers`
  - `pip install -U qwen-vl-utils accelerate bitsandbytes`
- Hugging Face (`HF`) downloads can be unstable due to network/rate limits.

## License

MIT. See `LICENSE`.
