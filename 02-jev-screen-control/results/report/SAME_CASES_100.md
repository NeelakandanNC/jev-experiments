Same 100 ScreenSpot-v2 instructions for every variant (those all of yolo+luna,yolo+luna@desc,omniparser+luna,omniparser+luna@desc answered; the seed-0 sample of 100).

| Variant | n | Accuracy | Text targets | Icon targets | ECE | Acc. on most-confident half | Pick latency (p50) |
|---|---|---|---|---|---|---|---|
| gpt-6-luna clicks x, y (no detector) | 100 | **96.0%** | 100.0% | 89.2% | 0.022 | 100.0% | 1.73 s |
| our YOLO + description → OneJev1.6-0.8B sees the screenshot | 100 | **68.0%** | 92.1% | 27.0% | 0.125 | 96.0% | 27.70 s |
| OmniParser + description → OneJev1.6-0.8B (local) | 100 | **65.0%** | 85.7% | 29.7% | 0.111 | 92.0% | 14.75 s |
| our YOLO + description → OneJev1.6-0.8B (local) | 100 | **71.0%** | 95.2% | 29.7% | 0.093 | 96.0% | 12.56 s |
| OmniParser → OneJev1.6-0.8B (local) | 100 | **59.0%** | 85.7% | 13.5% | 0.105 | 82.0% | 28.19 s |
| our YOLO → OneJev1.6-0.8B (local) | 100 | **62.0%** | 87.3% | 18.9% | 0.060 | 84.0% | 14.90 s |
| OmniParser + description → Sarvam Prism decision | 100 | **71.0%** | 85.7% | 45.9% | 0.208 | 92.0% | 0.42 s |
| OmniParser + description → gpt-6-luna decision | 100 | **85.0%** | 95.2% | 67.6% | 0.080 | 96.0% | 0.57 s |
| our YOLO + description → Sarvam Prism decision | 100 | **69.0%** | 88.9% | 35.1% | 0.121 | 84.0% | 0.40 s |
| our YOLO + description → gpt-6-luna chat picks id | 100 | **81.0%** | 96.8% | 54.1% | 0.082 | 98.0% | 1.51 s |
| our YOLO + description → gpt-6-luna decision | 100 | **78.0%** | 96.8% | 45.9% | 0.099 | 95.7% | 0.18 s |
| OmniParser → Sarvam Prism decision | 100 | **50.0%** | 63.5% | 27.0% | 0.188 | 70.0% | 0.41 s |
| our YOLO → Sarvam Prism decision | 100 | **52.0%** | 66.7% | 27.0% | 0.094 | 64.0% | 0.39 s |
| our YOLO → gpt-6-luna chat picks id | 100 | **76.0%** | 93.7% | 45.9% | 0.104 | 96.0% | 1.82 s |
| OmniParser → gpt-6-luna decision | 100 | **70.0%** | 87.3% | 40.5% | 0.081 | 87.8% | 0.33 s |
| our YOLO → gpt-6-luna decision | 100 | **72.0%** | 88.9% | 43.2% | 0.107 | 89.1% | 0.32 s |
