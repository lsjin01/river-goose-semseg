# Seven-class multispectral SegFormer KD

This experiment distills the best seven-class DINOv3 + Mask2Former checkpoint into
SegFormer-B1 and SegFormer-B0 students.

## Input and initialization

- Both teacher and student receive the same nine-channel source tile: seven sensor
  pages plus two precomputed ratio channels.
- A trainable spectral adapter maps the student's seven physical sensor channels to
  an ImageNet-normalized three-channel representation. The two ratio channels are
  deliberately excluded for `fusion_type=input`, matching the teacher contract.
- SegFormer starts from the official NVIDIA ADE20K pretrained B1/B0 weights. Only its
  seven-class classifier and the spectral adapter are newly initialized.
- The teacher is frozen and reconstructed from
  `best_epoch_075_miou_0.6207.pt`; it is never updated by KD.

## Objective and split policy

The student minimizes supervised cross entropy plus semantic-map KL divergence:

`loss = CE(student, GT) + KL(student / T, teacher / T) * T^2`, with `T=4`.

Ignore-label pixels (`255`) are removed from both losses. Training uses one random
768-pixel crop per training frame, within-image class-biased sampling probability
0.3, and horizontal flips. Validation runs every five epochs on complete validation
frames with 768-pixel tiles, stride 512, center-weighted score blending, and no
padding. The test split is not used during KD training.

## Queue

`tools/run_spectral_kd_queue.py` waits for the teacher's independent test evaluation
to finish, then trains `segformer_b1` followed by `segformer_b0` on physical GPU 1.
Each run uses batch size 2 and six accumulation steps (effective batch size 12), up
to 50 epochs, with early stopping after four non-improving validation checks.

Outputs are written to:

- `outputs/labeling_merged_algae_7class_kd/b1`
- `outputs/labeling_merged_algae_7class_kd/b0`
- `outputs/labeling_merged_algae_7class_kd/queue_state.json`

Monitor with:

```bash
cat outputs/labeling_merged_algae_7class_kd/queue_state.json
tail -f outputs/labeling_merged_algae_7class_kd/queue.log
```
