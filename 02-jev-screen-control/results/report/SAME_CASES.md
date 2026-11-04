Same 917 ScreenSpot-v2 instructions for every variant (those all of yolo+luna,yolo+luna@desc,omniparser+luna,omniparser+luna@desc answered).

| Variant | n | Accuracy | Text targets | Icon targets | ECE | Acc. on most-confident half | Pick latency (p50) |
|---|---|---|---|---|---|---|---|
| gpt-6-luna clicks x, y (no detector) | 914 | **96.7%** | 97.7% | 95.3% | 0.008 | 99.3% | 1.81 s |
| OmniParser + description → Sarvam Prism decision | 791 | **69.2%** | 89.7% | 41.4% | 0.206 | 93.4% | 0.41 s |
| OmniParser + description → gpt-6-luna decision | 917 | **80.7%** | 94.4% | 61.8% | 0.038 | 95.4% | 0.60 s |
| our YOLO + description → Sarvam Prism decision | 792 | **69.3%** | 91.9% | 38.9% | 0.079 | 87.1% | 0.40 s |
| our YOLO + description → gpt-6-luna chat picks id | 915 | **77.4%** | 96.4% | 51.0% | 0.100 | 95.6% | 1.39 s |
| our YOLO + description → gpt-6-luna decision | 917 | **74.2%** | 96.4% | 43.4% | 0.091 | 94.3% | 0.18 s |
| OmniParser → Sarvam Prism decision | 791 | **51.3%** | 73.2% | 21.7% | 0.201 | 78.1% | 0.41 s |
| our YOLO → Sarvam Prism decision | 792 | **53.4%** | 76.5% | 22.3% | 0.054 | 69.9% | 0.39 s |
| our YOLO → gpt-6-luna chat picks id | 913 | **72.7%** | 94.5% | 42.6% | 0.129 | 93.9% | 1.91 s |
| OmniParser → gpt-6-luna decision | 917 | **68.3%** | 88.5% | 40.3% | 0.051 | 92.5% | 0.34 s |
| our YOLO → gpt-6-luna decision | 917 | **67.0%** | 92.3% | 31.9% | 0.067 | 92.7% | 0.34 s |
