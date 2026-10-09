## Results


### Detector (`models/screenjev-yolo.pt`)

YOLO11n fine-tuned from COCO for 23 epochs at 640 px on cpu (4.63 h; 15 + 8 epochs: first dataset (MiniWoB <span class=alink> links unlabelled); then relabelled MiniWoB pages (pointer-cursor elements)). Train set: 4156 screenshots.

| Eval image size | Split | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|
| 640 | val | 94.2% | 77.7% | 91.9% | 89.3% |
| 640 | miniwob_suite | 71.5% | 52.3% | 90.5% | 60.3% |
| 960 | val | 95.4% | 79.7% | 94.0% | 91.4% |
| 960 | miniwob_suite | 64.2% | 39.6% | 75.4% | 59.6% |

![Detector training](detector_training.png)


Per-class mAP50-95 (val): button 89.6%, link 70.6%, text_input 89.8%, checkbox 70.0%, radio 60.9%, toggle 85.0%, dropdown 84.1%, slider 76.1%, tab 86.7%, icon 78.6%, back 85.9%, close 68.5%, menu 75.0%, search 71.6%, scrollbar 73.4%
