# PGTFormer

# 🔥 News
- *2024.04*: 🎉 This paper is accepted by IJCAI 2024

- Beyond Alignment: Blind Video Face Restoration via Parsing-Guided Temporal-Coherent Transformer (IJCAI 2024)

### Results report in VFHQ implementation.

| Metrics | Bicubic | EDVRM | BasicVSR | EDVRM-GAN | BasicVSR-GAN | DFDNet | GFP-GAN | GPEN |
|---------|---------|-------|----------|-----------|--------------|--------|---------|------|
| PSNR    | 26.842  | 29.457 (Red) | 29.472 (Red) | 26.682  | 25.813  | 25.178  | 25.978 | 26.672 |
| SSIM    | 0.7909  | 0.8428 (Red) | 0.8430 (Red) | 0.7638  | 0.741   | 0.7560  | 0.7723 | 0.7768 |
| LPIPS   | 0.4098  | 0.3288 | 0.3309 | 0.3076 (Red) | 0.3214 (Red) | 0.4008  | 0.3446 | 0.3607 |


### Results reported in our implementation.
| Method      | PSNR  | SSIM   | LPIPS | Deg  | LMD  | TLME | MSRL | | PSNR  | SSIM   | LPIPS | Deg  | LMD  | TLME | MSRL |
|-------------|-------|--------|-------|------|------|------|------|-|-------|--------|-------|------|------|------|------|
| BasicVSR++  | 26.70 | 0.7906 | 0.3086| 38.31| 2.89 | 6.97 | 24.15| | 26.17 | 0.7482 | 0.3594| 36.14| 2.39 | 7.09 | 23.91|
| HiFaceGAN   | 28.45 | 0.8297 | 0.2924| 34.02| 2.25 | 5.73 | 25.81| | 27.41 | 0.7926 | 0.3167| 32.74| 1.99 | 5.59 | 24.99|
| GFP-GAN     | 27.77 | 0.8252 | 0.2893| 31.70| 2.40 | 6.11 | 25.68| | 26.27 | 0.7864 | 0.3167| 30.14| 2.13 | 6.17 | 24.69|
| VQFR        | 25.59 | 0.7788 | 0.3003| 37.83| 2.99 | 7.41 | 23.60| | 25.33 | 0.7459 | 0.3134| 33.27| 2.40 | 7.05 | 23.04|
| Codeformer  | 28.71 | 0.8226 | 0.2460| 28.11| 1.97 | 5.82 | 26.32| | 27.88 | 0.8018 | 0.2738| 26.55| 1.74 | 5.60 | 25.54|
| Ours        | 30.74 | 0.8668 | 0.2095| 24.41| 1.63 | 4.20 | 28.16| | 29.66 | 0.8408 | 0.2230| 23.09| 1.35 | 4.09 | 27.33|


# Acknowledgement
This project is based on BasicSR. Some codes are brought from Codeformer. We also adopt the VFHQ dataset to train network.
