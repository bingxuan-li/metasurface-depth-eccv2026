# Licensing and publication checklist

Final approval for a license on newly written integration code is pending. Until
that is confirmed, this checkout is a release candidate rather than a published
open-source license grant. Existing upstream licenses remain in force.

The official [DAV2 license statement](https://github.com/DepthAnything/Depth-Anything-V2#license)
distinguishes Small (Apache-2.0) from Base/Large/Giant (CC-BY-NC-4.0). The selected
models were fine-tuned from metric-Hypersim weights; inspect and preserve the exact
upstream checkpoint terms when preparing their public model cards. Do not label
every model MIT or unrestricted commercial use merely because integration code
or simulator source uses MIT.

Before public publication:

- Confirm the integration-code license and author attribution for local modifications.
- Preserve simulator/DINOv2/DAV2/PromptDA notices and document PSF provenance.
- Finalize each checkpoint's license/model card and hosting URL.
- Confirm redistribution rights for any scientific images, labels or calibration assets.
- Scan the actual Git contents for credentials, private paths and large training files.
- Keep scientific limitations and validation claims consistent with executed tests.

The paper metadata is sourced from [arXiv:2503.15770](https://arxiv.org/abs/2503.15770).
This staging task has not published a GitHub repository or any public weight files.
