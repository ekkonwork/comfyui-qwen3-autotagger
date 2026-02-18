# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

- Fixed model loading flow for Qwen3-VL:
  - Removed incorrect fallback to `AutoModelForCausalLM` that could surface misleading `Unrecognized configuration class ... Qwen3VLConfig` errors.
  - Preserved and returned the primary model-loading error for easier debugging.
  - Added automatic retry without 4-bit quantization when 4-bit initialization fails.

## [0.1.0] - 2026-02-09

- Initial public release.
- Qwen3-VL auto-tagging with Adobe Stock style metadata.
- XMP embedding via exiftool.
- Headless/API workflow example.
- Node Docs and example workflows.
