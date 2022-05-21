## CIFAR-10

| DEVICE | FRAMEWORK | BATCH SIZE | TIME (PER TRAIN EPOCH) | TIME (PER VAL) |
|---|---|---|---|---|
| v3-8 (1 core) | torch | 128 | 11.8s | 50.7s |
| v3-8 (8 core) | torch | 1024 (128 x 8) | 12.6s | 7.9s (with mesh_reduce) |
| v3-8 (8 core) | torch | 128 (16 x 8) | 18.6s | 8.0s (with mesh_reduce) |
