## Results


### Detector (`models/screenjev-yolo.pt`)

YOLO11n fine-tuned from COCO for 15 epochs at 640 px on cpu (3.01 h). Train set: 4187 screenshots.

| Eval image size | Split | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|
| 640 | val | 92.5% | 75.6% | 90.0% | 86.2% |
| 640 | miniwob_suite | 70.4% | 51.4% | 89.7% | 67.2% |
| 960 | val | 94.3% | 78.5% | 92.3% | 90.3% |
| 960 | miniwob_suite | 65.0% | 40.6% | 82.3% | 64.3% |

![Detector training](detector_training.png)


Per-class mAP50-95 (val): button 88.2%, link 70.4%, text_input 89.6%, checkbox 68.9%, radio 60.0%, toggle 84.6%, dropdown 82.0%, slider 73.8%, tab 84.5%, icon 75.1%, back 81.0%, close 62.7%, menu 72.2%, search 69.2%, scrollbar 72.2%
