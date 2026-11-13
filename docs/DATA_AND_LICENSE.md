# Data, weights, and licensing

- HSI benchmark dataset: https://huggingface.co/datasets/hsi-workshop/referential-gesture-challenge
- SMPL-X model assets: https://smpl-x.is.tue.mpg.de/modellicense.html
- Challenge details: https://www.hsi-workshop.com/challenge

**Do not upload datasets, derived training/sample annotations, checkpoint files, output motions,
SMPL-X assets, or official test materials** until rights to redistribute them are verified.

The SMPL-X model license is for non-commercial research and states no redistribution without prior permission.
The Hugging Face dataset card does not itself establish a permissive redistribution license for
derived artifacts from this experiment. Get legal/owner confirmation for public weights.

RePointer-owned source code is released under the [MIT License](../LICENSE),
selected and approved by the owner for the public source release. This grants no
rights to third-party code, dataset content, model parameters or SMPL-X assets.
Source-only code hosting is separate from distribution of pretrained parameters.

The cleaned smoke workflow creates only invented fixtures. It may write synthetic
NPZ arrays and tiny training parameters under ignored `outputs/`; these are local
test artifacts and are excluded from source distributions. The optional learned
timing example requires an external private checkpoint directory. No asset downloads,
uploads, GitHub pushes or license selection are performed by the package.

Real `replay` execution stores all newly trained weights, per-sample results and
generated motions in a fresh external private directory. The public real-reproduction
report contains aggregate counts, environment versions and measured differences only.
It does not distribute dataset content or private parameters. No organizer submission
was uploaded by the real experiments.

The accepted historical M1 run remains external. Public `protocol/m1/` contains
only aggregate counts/status and evidence hashes; it includes no sample identifiers,
labels, motions, local paths or weights. Read-only `audit-m1` reports use the same
path-free boundary. This separation is documented in
[Repository Layout](REPOSITORY_LAYOUT.md) and [External Assets](EXTERNAL_ASSETS.md).
