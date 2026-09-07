# Licensing

Original integration code and project modifications are released under the
[MIT license](../LICENSE). Existing upstream licenses and copyright notices
remain in force: DINOv2, PromptDA and DAV2 source components are Apache-2.0;
the frozen V5 simulator and bundled PSF retain the original simulator MIT notice.
See [third-party notices](../THIRD_PARTY_NOTICES.md).

The official [DAV2 license statement](https://github.com/DepthAnything/Depth-Anything-V2#license)
distinguishes Small (Apache-2.0) from Base/Large/Giant (CC-BY-NC-4.0). The selected
models were fine-tuned from metric-Hypersim weights. Their model card and
per-checkpoint license notice preserve these upstream terms. Do not label
every model MIT or unrestricted commercial use merely because integration code
or simulator source uses MIT.

Redistribution:

- Preserve simulator/DINOv2/DAV2/PromptDA notices and document PSF provenance.
- Preserve each checkpoint's license and model card.
- Dataset distribution has separate terms; the code license does not cover data.
- Do not add credentials, private paths or original training checkpoints to Git.
- Keep scientific limitations and validation claims consistent with executed tests.

The paper metadata is sourced from [arXiv:2503.15770](https://arxiv.org/abs/2503.15770).
