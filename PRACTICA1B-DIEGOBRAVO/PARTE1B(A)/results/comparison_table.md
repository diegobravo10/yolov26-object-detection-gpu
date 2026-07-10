# YOLO Benchmark Comparison

Generated: 2026-07-10 00:11:18

## Results Table

| Model | Device | Precision | Avg FPS | Avg Infer (ms) | P95 Infer (ms) | Avg Total (ms) | Peak RAM (MB) | Peak VRAM (MB) | Frames |
|-------|--------|-----------|---------|---------------|---------------|---------------|---------------|----------------|--------|
| YOLOv12 | cpu | fp32 | 11.35 | 85.25 | 99.58 | 88.13 | 885.00 | 15.00 | 5 |
| YOLOv12 | cuda | fp32 | 77.20 | 9.92 | 11.00 | 12.95 | 871.00 | 16.00 | 5 |
| YOLOv26 | cpu | fp32 | 16.36 | 58.59 | 68.45 | 61.11 | 871.00 | 26.00 | 5 |
| YOLOv12 | cuda | fp32 | 95.78 | 8.80 | 11.19 | 10.44 | 871.00 | 46.00 | 30 |
| YOLOv12 | cuda | fp32 | 91.65 | 9.14 | 10.97 | 10.91 | 854.00 | 15.00 | 10 |
| YOLOv12 | cpu | fp32 | 13.74 | 71.27 | 85.01 | 72.77 | 935.98 | 11.53 | 338 |
| YOLOv12 | cuda | fp32 | 109.33 | 7.64 | 8.91 | 9.15 | 1094.69 | 30.83 | 338 |
| YOLOv12 | cpu | fp32 | 14.22 | 68.93 | 76.68 | 70.35 | 835.19 | 8.91 | 132 |
| YOLOv12 | cpu | fp32 | 14.33 | 68.35 | 80.50 | 69.78 | 822.26 | 21.54 | 338 |
| YOLOv12 | cpu | fp32 | 14.22 | 68.82 | 80.02 | 70.30 | 792.03 | 20.55 | 338 |
| YOLOv12 | cpu | fp32 | 14.18 | 69.04 | 80.29 | 70.51 | 819.38 | 20.95 | 338 |
| YOLOv12 | cpu | fp32 | 14.52 | 67.42 | 78.13 | 68.88 | 893.07 | 15.89 | 338 |
| YOLOv12 | cuda | fp32 | 107.45 | 7.71 | 8.42 | 9.31 | 1105.00 | 34.02 | 338 |
| YOLOv26 | cpu | fp32 | 22.32 | 43.86 | 50.78 | 44.80 | 877.64 | 19.93 | 338 |
| YOLOv26 | cuda | fp32 | 115.06 | 7.66 | 8.81 | 8.69 | 1198.83 | 33.61 | 338 |
| YOLOv26 | cpu | fp32 | 26.81 | 35.83 | 41.71 | 37.30 | 950.28 | 19.13 | 338 |
| YOLOv26 | cuda | fp32 | 130.75 | 6.02 | 7.01 | 7.65 | 1153.67 | 32.86 | 338 |

## Speedup Analysis

- **YOLOv12** GPU vs CPU speedup: **6.80x** (11.35 → 77.20 FPS)
- **YOLOv26** GPU vs CPU speedup: **7.03x** (16.36 → 115.06 FPS)

### Cross-model comparison

- CPU FPS diff (YOLOv12 vs YOLOv26): -5.02 FPS
- GPU FPS diff (YOLOv12 vs YOLOv26): -37.86 FPS

### Summary

- Fastest on CPU: **YOLOv26**
- Fastest on GPU: **YOLOv26**
- Lowest VRAM:    **YOLOv12**
- Lowest latency: **YOLOv26 (cuda)**
