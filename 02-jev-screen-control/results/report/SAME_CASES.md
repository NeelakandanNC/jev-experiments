Same 917 ScreenSpot-v2 instructions for every variant (those all of yolo+luna,yolo+luna@desc,omniparser+luna,omniparser+luna@desc answered).

| Variant | n | Accuracy | Text targets | Icon targets | ECE | Acc. on most-confident half | Pick latency (p50) |
|---|---|---|---|---|---|---|---|
| gpt-6-luna clicks x, y (no detector) | 914 | **96.7%** | 97.7% | 95.3% | 0.008 | 99.3% | 1.81 s |
| our YOLO + description → OneJev1.6-0.8B sees the screenshot | 100 | **68.0%** | 92.1% | 27.0% | 0.125 | 96.0% | 27.70 s |
| OmniParser + description → OneJev1.6-0.8B (local) | 100 | **65.0%** | 85.7% | 29.7% | 0.111 | 92.0% | 14.75 s |
| our YOLO + description → OneJev1.6-0.8B (local) | 100 | **71.0%** | 95.2% | 29.7% | 0.093 | 96.0% | 12.56 s |
| OmniParser → OneJev1.6-0.8B (local) | 100 | **59.0%** | 85.7% | 13.5% | 0.105 | 82.0% | 28.19 s |
| our YOLO → OneJev1.6-0.8B (local) | 100 | **62.0%** | 87.3% | 18.9% | 0.060 | 84.0% | 14.90 s |
| OmniParser + description → Sarvam Prism decision | 917 | **68.7%** | 89.5% | 40.0% | 0.206 | 92.6% | 0.41 s |
| OmniParser + description → gpt-6-luna decision | 917 | **80.7%** | 94.4% | 61.8% | 0.038 | 95.4% | 0.60 s |
| our YOLO + description → Sarvam Prism decision | 917 | **68.9%** | 91.5% | 37.7% | 0.085 | 86.5% | 0.40 s |
| our YOLO + description → gpt-6-luna chat picks id | 915 | **77.4%** | 96.4% | 51.0% | 0.100 | 95.6% | 1.39 s |
| our YOLO + description → gpt-6-luna decision | 917 | **74.2%** | 96.4% | 43.4% | 0.091 | 94.3% | 0.18 s |
| OmniParser → Sarvam Prism decision | 917 | **50.5%** | 72.2% | 20.5% | 0.199 | 76.2% | 0.41 s |
| our YOLO → Sarvam Prism decision | 917 | **53.5%** | 75.8% | 22.9% | 0.062 | 70.5% | 0.40 s |
| our YOLO → gpt-6-luna chat picks id | 913 | **72.7%** | 94.5% | 42.6% | 0.129 | 93.9% | 1.91 s |
| OmniParser → gpt-6-luna decision | 917 | **68.3%** | 88.5% | 40.3% | 0.051 | 92.5% | 0.34 s |
| our YOLO → gpt-6-luna decision | 917 | **67.0%** | 92.3% | 31.9% | 0.067 | 92.7% | 0.34 s |
