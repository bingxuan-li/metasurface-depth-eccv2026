# Third-party provenance

- `website/src/components/ui/select.tsx`: retained shadcn/ui select primitive,
  MIT (Copyright 2023 shadcn). The full notice is preserved in
  `licenses/SHADCN-MIT.txt`. Installed website dependencies retain their own terms.

- `src/metasurface_depth/_vendor/`: the experiment's DINOv2 implementation,
  originally from Meta. Copyright headers are retained; Apache-2.0 text is in
  `licenses/DINOV2-APACHE-2.0.txt`. Module imports were made package-relative.
- `src/metasurface_depth/models/`: experiment-specific PromptDA/DAV2-derived DPT
  components. Depth Anything V2 source headers are retained. The upstream
  [PromptDA](https://github.com/DepthAnything/PromptDA) and
  [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) repositories
  identify Apache-2.0 source-code licensing; the complete Apache-2.0 text is
  included in `licenses/DINOV2-APACHE-2.0.txt` and also applies to these upstream
  source components. Original project modifications are distributed under the
  root MIT license without replacing these upstream terms.
- `src/metasurface_depth/simulator/_kernel.py`, `_occlusion.py`, and the PSF asset:
  frozen from the project's V5 simulator. The existing MIT notice naming Yunxiang
  is retained verbatim in `licenses/SIMULATOR-MIT.txt` and covers the frozen
  simulator and bundled calibration resource.
- `augmentation.py` extracts only the original sensor-augmentation class. It
  does not import or distribute the old training entry point, telemetry setup,
  credentials, machine-specific paths, scheduler jobs or experiment logs.

Model weights have separate upstream terms. This notice is not a replacement
for their licenses and does not grant commercial rights to restricted weights.
See `docs/LICENSING.md` and preserve all notices when redistributing.
