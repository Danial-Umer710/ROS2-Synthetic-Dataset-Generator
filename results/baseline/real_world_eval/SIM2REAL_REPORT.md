# Sim2Real Baseline Gap — Real UR16e Camera Evaluation

Real test set: `ABC-iRobotics/LabWare_Items`, `instance_segmentation` config, official **test** split
(282 images @ 1280x800, 1,592 instances). Annotations converted to YOLOv8-OBB by
`scripts/prep_real_test_set.py` (minAreaRect over visible instance-mask pixels, same
visibility convention as the synthetic labels). Evaluation: imgsz 640, conf 0.001, NMS IoU 0.7.
Artifacts: `results/baseline/real_world_eval/` and `results/blenderproc/real_world_eval/`
(curves, confusion matrices, `metrics.json`, per-frame prediction overlays in `predictions/`).

## Headline numbers

| Model | Training data | Real mAP50 | Real mAP50-95 | Synthetic test mAP50-95 | Gap |
|---|---|---:|---:|---:|---:|
| `weights/best.pt` (clean baseline) | 800 clean synthetic | 49.38% | **37.28%** | 88.29% | -51.0 pts |
| `weights/blenderproc/best.pt` (distractors + loose tubes) | 800 synthetic w/ distractors | 66.30% | **48.91%** | 87.92% | -39.0 pts |

## Baseline model (`weights/best.pt`) per class — real test
| Class | Instances | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Tube_Holder | 282 | 0.899 | 0.755 | 0.876 | 0.710 |
| Glass_Tube | 468 | 0.256 | 0.068 | 0.048 | 0.020 |
| Slide_Holder | 560 | 0.443 | 0.420 | 0.417 | 0.272 |
| Eppendorf_Rack | 282 | 0.564 | 0.614 | 0.634 | 0.490 |

## Distractor model (`weights/blenderproc/best.pt`) per class — real test
| Class | Instances | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Tube_Holder | 282 | 0.960 | 0.759 | 0.938 | 0.750 |
| Glass_Tube | 468 | 0.385 | 0.573 | 0.434 | 0.293 |
| Slide_Holder | 560 | 0.660 | 0.402 | 0.552 | 0.352 |
| Eppendorf_Rack | 282 | 0.622 | 0.723 | 0.728 | 0.562 |

## Findings
- Large Sim2Real gap for the clean-baseline model: synthetic test mAP50-95 88.29% -> real 37.28%.
- Distractor model halves the gap: real mAP50-95 48.91% (+11.6 pts), mAP50 66.30% (+16.9 pts).
- Glass_Tube is the weakest class: real tubes are Falcon-style conical tubes lying loose, while the baseline only ever saw thin glass tubes upright inside TubeHolder holes. Distractor model's loose-tube training raised real Glass_Tube mAP50 4.8% -> 43.4%.
- Known label-space caveat: real 'Tube' (conical centrifuge tube) differs physically from synthetic Tube.stl (thin glass test tube); real scenes also contain unannotated clutter (syringes, slides, caps) that counts as false positives.
- Baseline model systematically misclassifies real loose tubes as Slide_Holder; distractor model hallucinates Glass_Tube detections inside empty TubeHolder holes (synthetic bias: holes always contained tubes).

## Caveats
- These are real-camera detections against auto-generated instance-segmentation ground truth
  (Label Factory annotations "must be independently reviewed" per the dataset card).
- `Glass_Tube` mapping: real `Tube` = Falcon-style conical centrifuge tube; our synthetic
  `Tube.stl` = thin glass test tube. The class id mapping follows Károly's naming, but the
  visual domain shift for this class is larger than for the others.
- Unannotated scene clutter (syringes, slides, caps, boxes) counts as false positives in mAP.
