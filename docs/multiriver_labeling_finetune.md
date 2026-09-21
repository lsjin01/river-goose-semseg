# Multi-river teacher to Labeling_Data fine-tuning

This stage runs after the current SegFormer-B0 experiment finishes. It initializes
the Labeling_Data teacher from the DINOv3 + Mask2Former model previously trained on
the multi-river `goose_data_cat2` dataset:

`outputs_old/baseline_100ep/baseline_cat2_100ep/best.pt`

It is transfer learning, not checkpoint resume. Optimizer, scaler, scheduler, epoch,
and early-stopping state all start fresh.

## Transfer boundary

The source is a 3-channel, 12-class model. The target consumes nine stored channels
(seven sensor pages plus two ratio channels) and predicts the seven-class merged
algae taxonomy. Adding the spectral input adapter shifts the internal module indices;
the checkpoint loader remaps those indices before shape matching.

- Reused: 894 compatible tensors, including DINOv3 and the compatible Mask2Former
  decoder (340,355,850 parameters).
- New: five spectral-adapter tensors and the two seven-class classifier tensors.
- Deliberately not reused: the source 12-class classifier.

This transfers 99.9993% of the target model parameters by parameter count while
keeping the input and output contracts correct.

## Target protocol

- Data: `data/labeling_merged_algae_7class_v2_geo`
- Scene/management-group split: 2,790 train, 572 validation, 623 test frames
- GPU: physical GPU 1
- Effective batch size: 2 x 6 accumulation = 12
- Maximum epochs: 100
- Full-frame overlap validation every five epochs
- Early stopping: four validation checks without improvement
- Test split: never evaluated during training

The best fine-tuned teacher will later be evaluated once on the independent test
split and then used to distill new SegFormer-B1 and SegFormer-B0 students.

Monitor the waiting/running queue with:

```bash
cat outputs/labeling_merged_algae_7class_multiriver_ft_gpu1/queue_state.json
tail -f outputs/labeling_merged_algae_7class_multiriver_ft_gpu1/queue.log
```
