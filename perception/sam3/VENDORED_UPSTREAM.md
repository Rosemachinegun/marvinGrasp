# Vendored SAM3 source

This directory vendors the image-inference runtime subset of the source code
from:

- Repository: https://github.com/facebookresearch/sam3
- Commit: `86ed77094094e5cabb16b0414ec60c5ba9ce0a0f`

The model checkpoint is intentionally not duplicated here. MarvinGrasp loads it
from `perception/models/sam3.pt`.

Upstream demos, media assets, notebooks, evaluation tools, agent code, and
training-only code are intentionally omitted. The retained modules cover the
`build_sam3_image_model` and `Sam3Processor` path used by MarvinGrasp.

The upstream license is preserved in `LICENSE`.
