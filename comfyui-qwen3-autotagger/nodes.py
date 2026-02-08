import base64
import io
import json
import os
import re
import shutil
import subprocess
import time
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
except Exception:  # pragma: no cover - fallback for older transformers
    Qwen3VLForConditionalGeneration = None
    AutoProcessor = None
    AutoModelForCausalLM = None
    BitsAndBytesConfig = None

try:
    from qwen_vl_utils import process_vision_info
except Exception as e:
    process_vision_info = None
    _qwen_import_error = e
else:
    _qwen_import_error = None

try:
    import folder_paths
except Exception:
    folder_paths = None

MODEL_CACHE = {}
LOCAL_MODEL_SUBDIRS = ("LLM", "llm")

DEFAULT_PROMPT = (
    "You are an Adobe Stock metadata generator.\n"
    "Return ONLY a valid JSON object.\n"
    "Structure: {\"title\": \"English title max 150 chars\", \"keywords\": [\"tag1\", \"tag2\", \"tag3\"]}\n"
    "Rules:\n"
    "1. GENERATE ~60 KEYWORDS. Include synonyms, visual details, concepts.\n"
    "2. PREFER SINGLE WORDS where possible.\n"
    "3. Output raw JSON only.\n"
)


def _tensor_to_pil_and_base64(image_tensor: torch.Tensor) -> Tuple[Image.Image, str]:
    if image_tensor.dtype != torch.float32:
        image_tensor = image_tensor.float()
    image_np = image_tensor.clamp(0, 1).cpu().numpy()
    image_np = (image_np * 255.0).round().astype(np.uint8)
    pil = Image.fromarray(image_np)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return pil, f"data:image/png;base64,{b64}"


def _extract_and_fix_json(text: str):
    text = text.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    start = text.find("{")
    if start != -1:
        json_fragment = text[start:]
        if "}" not in json_fragment and '"keywords": [' in json_fragment:
            fixed_json = json_fragment.rstrip(", ") + '"]}'
            fixed_json = fixed_json.replace('""', '"')
            try:
                return json.loads(fixed_json)
            except Exception:
                pass

    return None


def _clean_split_and_limit(keywords_list, limit=50) -> List[str]:
    final_list = []
    for item in keywords_list:
        clean_item = str(item).replace("_", " ").replace("-", " ")
        words = clean_item.split(" ")
        for w in words:
            w = w.strip().lower()
            if len(w) > 2:
                final_list.append(w)

    seen = set()
    unique_list = []
    for x in final_list:
        if x not in seen:
            unique_list.append(x)
            seen.add(x)

    return unique_list[:limit]


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_models_root() -> str:
    if folder_paths is not None and hasattr(folder_paths, "models_dir"):
        return folder_paths.models_dir
    if folder_paths is not None and hasattr(folder_paths, "get_folder_paths"):
        try:
            checkpoints = folder_paths.get_folder_paths("checkpoints")
            if checkpoints:
                return os.path.dirname(os.path.normpath(checkpoints[0]))
        except Exception:
            pass
    return os.path.join(os.getcwd(), "models")


def _list_local_models() -> List[str]:
    base = _get_models_root()
    choices: List[str] = []
    if not base or not os.path.isdir(base):
        return choices
    for sub in LOCAL_MODEL_SUBDIRS:
        subdir = os.path.join(base, sub)
        if not os.path.isdir(subdir):
            continue
        for name in sorted(os.listdir(subdir)):
            full = os.path.join(subdir, name)
            if os.path.isdir(full):
                choices.append(os.path.join(sub, name))
    return choices


def _get_local_model_choices() -> List[str]:
    choices = _list_local_models()
    return ["(manual)"] + choices if choices else ["(manual)"]


def _resolve_model_reference(model_id: str, auto_download: bool, local_model: str, local_model_path: str) -> str:
    if auto_download:
        return (model_id or "").strip()

    if local_model and local_model != "(manual)":
        base = _get_models_root()
        if base:
            return os.path.join(base, local_model)
        return local_model

    if local_model_path:
        return os.path.expanduser(local_model_path.strip())

    return (model_id or "").strip()


def _get_model_and_processor(
    model_ref: str,
    min_pixels: int,
    max_pixels: int,
    load_in_4bit: bool,
    allow_download: bool,
):
    device = _get_device()
    cache_key = (model_ref, min_pixels, max_pixels, load_in_4bit, device.type, allow_download)
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    if AutoProcessor is None:
        raise RuntimeError("transformers is not available in this environment")
    if process_vision_info is None:
        raise RuntimeError(f"qwen-vl-utils is not available: {_qwen_import_error}")

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    device_map = "auto" if device.type == "cuda" else None

    quant_config = None
    if load_in_4bit:
        if device.type != "cuda":
            print("4-bit quantization requires CUDA; loading without quantization")
        elif BitsAndBytesConfig is None:
            print("bitsandbytes is not available; loading without 4-bit quantization")
        else:
            try:
                quant_config = BitsAndBytesConfig(load_in_4bit=True)
            except Exception:
                quant_config = None

    model = None
    local_files_only = not allow_download

    if Qwen3VLForConditionalGeneration is not None:
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_ref,
                torch_dtype=dtype,
                device_map=device_map,
                quantization_config=quant_config,
                local_files_only=local_files_only,
            )
        except Exception:
            model = None

    if model is None:
        if AutoModelForCausalLM is None:
            raise RuntimeError("Unable to import Qwen3 classes or AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            quantization_config=quant_config,
            local_files_only=local_files_only,
        )

    if device_map is None:
        model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_ref,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        local_files_only=local_files_only,
    )

    MODEL_CACHE[cache_key] = (model, processor, device)
    return model, processor, device


def _save_with_xmp(pil: Image.Image, title: str, keywords: List[str], output_dir: str, prefix: str, index: int, fmt: str):
    if not output_dir:
        if folder_paths is not None:
            output_dir = folder_paths.get_output_directory()
        else:
            output_dir = os.getcwd()

    os.makedirs(output_dir, exist_ok=True)
    safe_prefix = re.sub(r"[^a-zA-Z0-9_-]", "_", prefix) if prefix else "autotag"
    filename = f"{safe_prefix}_{index:05d}.{fmt}"
    path = os.path.join(output_dir, filename)

    save_kwargs = {}
    if fmt.lower() in {"jpg", "jpeg"}:
        save_kwargs["quality"] = 95
        save_kwargs["subsampling"] = 0
        fmt = "JPEG"
    elif fmt.lower() == "webp":
        save_kwargs["quality"] = 95
        fmt = "WEBP"
    else:
        fmt = "PNG"

    pil.save(path, format=fmt, **save_kwargs)

    if shutil.which("exiftool") is None:
        print("exiftool not found in PATH; skipping XMP embedding")
        return path

    if len(title) > 199:
        title = title[:199]
    keywords_str = ", ".join(keywords)

    cmd = [
        "exiftool",
        "-overwrite_original",
        "-codedcharacterset=utf8",
        "-m",
        f"-XMP-dc:Title={title}",
        f"-XMP-dc:Description={title}",
        "-sep",
        ", ",
        f"-XMP-dc:Subject={keywords_str}",
        path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return path


class Qwen3VLAutoTagger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "system_prompt": (
                    "STRING",
                    {
                        "default": DEFAULT_PROMPT,
                        "multiline": True,
                        "tooltip": "System prompt that enforces JSON output.",
                    },
                ),
                "max_keywords": (
                    "INT",
                    {"default": 50, "min": 5, "max": 200, "step": 1, "tooltip": "Keep up to N keywords."},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 600, "min": 32, "max": 2048, "step": 1, "tooltip": "Generation length."},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Sampling temperature."},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Nucleus sampling."},
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {
                        "default": 1.15,
                        "min": 0.5,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Discourage repeats.",
                    },
                ),
                "attempts": (
                    "INT",
                    {"default": 3, "min": 1, "max": 5, "step": 1, "tooltip": "Retry count if JSON is invalid."},
                ),
                "min_pixels": (
                    "INT",
                    {
                        "default": 256 * 28 * 28,
                        "min": 64 * 28 * 28,
                        "max": 1024 * 28 * 28,
                        "step": 28 * 28,
                        "tooltip": "Minimum vision resolution budget.",
                    },
                ),
                "max_pixels": (
                    "INT",
                    {
                        "default": 756 * 756,
                        "min": 256 * 256,
                        "max": 1536 * 1536,
                        "step": 1,
                        "tooltip": "Maximum vision resolution budget.",
                    },
                ),
                "allow_resize": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Allow processor to resize images to valid patch sizes.",
                    },
                ),
                "model_id": (
                    "STRING",
                    {
                        "default": "Qwen/Qwen3-VL-8B-Instruct",
                        "tooltip": "HF model id (used when auto_download is enabled).",
                    },
                ),
                "auto_download": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Allow downloading the model from Hugging Face on first run.",
                    },
                ),
                "local_model": (
                    _get_local_model_choices(),
                    {
                        "default": "(manual)",
                        "tooltip": "Pick a local model under models/LLM (or use manual path).",
                    },
                ),
                "local_model_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Full path to a local model folder (used when local_model is '(manual)').",
                    },
                ),
                "load_in_4bit": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Enable 4-bit quantization (CUDA only)."},
                ),
                "write_xmp": ("BOOLEAN", {"default": False, "tooltip": "Embed XMP metadata via exiftool."}),
                "output_dir": (
                    "STRING",
                    {"default": "", "tooltip": "Save directory. Empty uses ComfyUI output dir."},
                ),
                "output_format": (
                    "STRING",
                    {"default": "jpg", "tooltip": "Output format: jpg, png, or webp."},
                ),
                "file_prefix": (
                    "STRING",
                    {"default": "qwen3_autotag", "tooltip": "Output filename prefix."},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "title", "keywords", "json", "xmp_paths")
    FUNCTION = "tag"
    CATEGORY = "Metadata"

    def tag(
        self,
        image,
        system_prompt,
        max_keywords,
        max_new_tokens,
        temperature,
        top_p,
        repetition_penalty,
        attempts,
        min_pixels,
        max_pixels,
        allow_resize,
        model_id,
        auto_download,
        local_model,
        local_model_path,
        load_in_4bit,
        write_xmp,
        output_dir,
        output_format,
        file_prefix,
    ):
        model_ref = _resolve_model_reference(model_id, auto_download, local_model, local_model_path)
        if not model_ref:
            raise RuntimeError("Model is not set. Provide model_id or select a local model.")

        using_local_selection = (not auto_download) and (
            (local_model and local_model != "(manual)") or (local_model_path and local_model_path.strip())
        )
        if using_local_selection and not os.path.isdir(model_ref):
            raise RuntimeError(f"Local model path not found: {model_ref}")

        model, processor, _device = _get_model_and_processor(
            model_ref,
            min_pixels,
            max_pixels,
            load_in_4bit,
            allow_download=bool(auto_download),
        )

        batch_size = image.shape[0]
        titles = []
        keywords_out = []
        json_out = []
        xmp_paths = []

        for idx in range(batch_size):
            pil, b64 = _tensor_to_pil_and_base64(image[idx])

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": b64,
                            "max_pixels": int(max_pixels),
                            "min_pixels": int(min_pixels),
                        },
                        {"type": "text", "text": system_prompt},
                    ],
                }
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            try:
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    do_resize=bool(allow_resize),
                )
            except TypeError:
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )

            if hasattr(model, "device"):
                inputs = inputs.to(model.device)

            title = ""
            final_keywords = []
            raw_json = ""

            for attempt in range(attempts):
                try:
                    do_sample = attempt > 0 and temperature > 0
                    gen_kwargs = {
                        "max_new_tokens": int(max_new_tokens),
                        "repetition_penalty": float(repetition_penalty),
                        "do_sample": bool(do_sample),
                    }
                    if do_sample:
                        gen_kwargs["temperature"] = float(temperature)
                        gen_kwargs["top_p"] = float(top_p)

                    with torch.inference_mode():
                        generated_ids = model.generate(**inputs, **gen_kwargs)

                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]

                    data = _extract_and_fix_json(output_text)
                    if data and data.get("title") and data.get("keywords"):
                        title = data["title"]
                        final_keywords = _clean_split_and_limit(data["keywords"], limit=max_keywords)
                        raw_json = json.dumps({"title": title, "keywords": final_keywords}, ensure_ascii=False)
                        if len(final_keywords) < 5 and attempt < (attempts - 1):
                            continue
                        break
                except Exception as e:
                    if attempt == attempts - 1:
                        if "out of memory" in str(e).lower() and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        print(f"Tagger error: {e}")
                    time.sleep(1)

            titles.append(title)
            keywords_out.append(", ".join(final_keywords))
            json_out.append(raw_json)

            if write_xmp:
                path = _save_with_xmp(pil, title, final_keywords, output_dir, file_prefix, idx, output_format)
                xmp_paths.append(path)

        titles_str = "\n".join(titles)
        keywords_str = "\n".join(keywords_out)
        json_str = "\n".join(json_out)
        xmp_paths_str = "\n".join(xmp_paths) if xmp_paths else ""

        return (image, titles_str, keywords_str, json_str, xmp_paths_str)


NODE_CLASS_MAPPINGS = {
    "Qwen3VLAutoTagger": Qwen3VLAutoTagger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VLAutoTagger": "Qwen3 VL AutoTagger",
}
