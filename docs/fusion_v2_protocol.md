# Fusion v2: leakage audit and training protocol

## Frozen split

The original split placed all 17 tasks across training and evaluation splits.
The 795 original image files have distinct SHA-256 hashes. An additional coarse
24x24, per-channel-standardized RGB thumbnail correlation screen found no
cross-task same-sensor pairs above 0.97. This heuristic does not certify scene independence.

P1 EXIF GPS shows repeated locations in different tasks. Tasks are merged into
connected components whenever any camera positions are within 1 km, across dates.
For example, task31/48/9 overlap; task50/52/7/8 share a nearby region;
task10/23 are also coupled by the conservative 1 km rule. All images and all
tiles of a component stay in one split. Altum TIFFs have no GPS/time metadata;
their tasks remain indivisible, but Altum location independence and cross-sensor
scene independence cannot be certified without the missing acquisition mapping.

| Split | Images | Altum | P1 | Evaluation tiles |
|---|---:|---:|---:|---:|
| Train | 539 | 216 | 323 | random native-grid crops |
| Validation | 142 | 69 | 73 | 4,278 |
| Test | 114 | 41 | 73 | 6,670 |

All source paths, tasks, exact image hashes and identified P1 geographic groups
are disjoint across splits. Group assignment uses only identities and label
coverage, not model predictions or metrics. The rare algae4-rich task53 stays in
train. Algae4 raster pixels: train 5,408,125; validation 52,326; test 27,297.

Algae0 exists in only one independent P1 region, which stays in train. The model
still predicts 11 classes. Primary evaluation mIoU uses the fixed 10 GT-supported
classes (all except algae0), consistently for all five models. Confusion matrices
and all 11 class rows remain available; unsupported classes must be reported as
unmeasurable, not successes. Historical 11-class scores on the old split are not
directly comparable. Validation turbid has only one annotated image; per-class
results have substantial sampling uncertainty.

Manifest: `data/fusion_v2_geo/manifest.json`.
SHA-256: `c621efd7254d2c37c8b1e5de4a084086ae6fccf14af9ae48e96bfc4991e96826`.
Audits: `split_audit.json` and `verification.json` in the same directory.
`data/fusion_v2` and `data/fusion_v2_grouped` are rejected preliminary candidates,
not training inputs. Their small metadata files are retained for traceability.

## Inputs and model changes

- Preserve all seven uint16 TIFF pages, without claiming that pages 5/6 are
  thermal/NoData. Their physical identity remains unverified. P1 uses B/G/R slots
  and zeros for unavailable pages and indices.
- Estimate per-sensor per-band 1st/99th percentiles and standard deviations from
  **train images only**. Scale by max(percentile range, standard deviation, 1).
  Apply invertible `asinh` compression to normalized values, preventing rare
  deviations in near-constant pages from dominating; no hard clipping.
- Nine loader slots: seven normalized pages plus two raw-page ratio proxies
  `(page3-page2)/(page3+page2+eps)` and `(page3-page4)/(page3+page4+eps)`.
  Ratios are calculated before normalization, and zeroed for P1. These are
  NDVI/NDRE candidates, **not calibrated physical indices** until band mapping is verified.
- A uses seven pages, B uses all nine slots. C/D/E use three RGB-position pages
  plus the four extra source pages through a spectral branch.
- Actual pretrained ViT parameters alone use encoder LR by object identity;
  wrappers no longer move spatial adapters into the 40x lower LR group.
- D gates depend on both RGB and spectral features. New feature fusion paths
  start as zero residuals. Spatial spectral stems downsample early to control cost.
- Train crops are extracted at COCO annotation-grid resolution, before resize of
  a source crop where P1 source/annotation dimensions differ. Altum retains its
  native pixel grid. A 70% class-first sampling branch selects a class, an image,
  then a labeled pixel; the remainder uses ordinary image-based random crops.
- Evaluation covers the entire annotation grid with nonoverlapping 768x768 tiles,
  pads only edge tiles, and ignores padding in metrics. No scales/hflip/TTA.

## Five declared experiments

| Config under config/fusion_v2 | Model |
|---|---|
| a_input.yaml | learned 7-to-3 input fusion |
| b_indices.yaml | learned 9-to-3 fusion with raw-page ratio proxies |
| c_late.yaml | independent spectral branch, final-level residual fusion |
| d_gated.yaml | four-level RGB-conditioned gated fusion |
| e_cross_attention.yaml | RGB queries and pooled spectral key/value tokens |

Every experiment starts from external DINOv3/Mask2Former pretraining, not old
task-trained checkpoints. Seed 42, identical independent loader generators,
batch 4 across GPUs 0/1, gradient accumulation 3, tile size 768, LR 4e-5,
ViT LR 1e-6, weight decay .02, polynomial schedule, warmup 200 updates.
Maximum 100 epochs. Full validation every 5 epochs; stop after four validation
checks without improvement. Best checkpoint is selected using validation only.
Retain one best and one latest checkpoint per experiment. Existing historical
checkpoints are not deleted. The final partial accumulation window is normalized
by its actual number of microbatches.

The queue trains A through E, then evaluates each validation-selected best on
the frozen test set, reporting overall/Altum/P1 and per-class results. It freezes
source/config/manifest hashes, uses a lock, records failures, and stops rather
than running through a failure. Restart uses latest checkpoints and restores
patience from validation logs; it is not guaranteed bit-identical to uninterrupted
training because full RNG/loader states are not checkpointed.

## Commands

```bash
# Audit the existing frozen split
/opt/conda/envs/goose/bin/python tools/audit_fusion_v2_split.py

# Start or resume the queue (use its systemd service if already launched)
/opt/conda/envs/goose/bin/python tools/run_fusion_v2_queue.py

# Queue status and first-model log
tail -n 10 outputs/fusion_v2/queue_status.jsonl
tail -f outputs/fusion_v2/a_input_train.log
systemctl --user status dummdumm-fusion-v2-queue.service
```

Validation metrics: `outputs/fusion_v2/<experiment>/epoch_metrics.csv`.
Final test: `test_no_tta.json` in each experiment directory.
Final comparison: `outputs/fusion_v2/summary.json` (created after all test runs).
Logs and training state are local/gitignored; scripts/configs/protocol can be committed.
