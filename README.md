<div align="center">

<h1>Beyond Alignment: Blind Video Face Restoration via Parsing-Guided Temporal-Coherent Transformer</h1>
<div>
    <a href='https://kepengxu.github.io/' target='_blank'>Kepeng Xu</a>&emsp;
    <a href='https://icecherylxuli.github.io/' target='_blank'>Li Xu</a>&emsp;
    <a href='' target='_blank'>Gang He et al</a>
</div>
<div>
    Xidian University,Southwest University of Science and Technology&emsp; 
</div>

<div>
    <strong>IJCAI 2024</strong>
</div>

<div>
    <h4 align="center">
        <a href="https://kepengxu.github.io/projects/pgtformer/" target='_blank'>
        <img src="https://img.shields.io/badge/🐳-Project%20Page-blue">
        </a>
    </h4>
</div>


<p align="center">
  <img src="./assets/output.gif" alt="showcase">
  <br>
  🔥 For more results, visit our <a href="https://kepengxu.github.io/projects/pgtformer/"><strong>project page</strong>,<a href="https://github.com/kepengxu/PGTFormer/blob/main/MoreVideoResults.md"><strong>gif page</strong></a> 🔥
  <br>
  ⭐ If you found this project helpful to your projects, please help star this repo. Thanks! 🤗
</p>

</div>


if you have any questions; please send message to me via Wechat or email: kepengxu11@gmail.com

<p align="center">
  <img src="./assets/vx.png" alt="Wechat" width="200">
</p>


# Update
- **2024.09**: Update Windows inference script.(Thanks @KFK121)
- **2024.08**: 🤗 This model has been successfully pushed to the Hugging Face **kepeng/pgtformer-base**, making it accessible to researchers and developers worldwide. Feel free to explore, test, and utilize it
- **2024.08**: We released the initial version of the inference code and models. Stay tuned for continuous updates!
- **2024.04**: This repo is created!


# Getting Started

## Dependencies and Installation

required packages in `requirements`
```
# git clone this repository
git clone https://github.com/kepengxu/PGTFormer
cd PGTFormer

# create new anaconda env
conda create -n pgtformer python=3.8 -y
conda activate pgtformer

# install python dependencies
conda install -c conda-forge dlib
conda install -c conda-forge ffmpeg
```

## Quick Inference

### Download Pre-trained Models
All pretrained models can also be automatically downloaded during the first inference.
You can also download our pretrained models from [Google Drive](https://drive.google.com/file/d/1DFwfPpiIxqjd-PrQ7zoKlVF4HqTM0Ops/view?usp=sharing),[BaiduYun](https://pan.baidu.com/s/1pN0u6ITT-JUFg9-PhgUkrg?pwd=pgtf),[Huggingface](https://huggingface.co/kepeng/pgtformer-base) to the `weights` folder.


### Prepare Testing Data
We provide a example in `assets/inputdemovideo.mp4`. If you would like to test your own face videos, place them in the same folder.




### Inference

**[Note]** 🚀🚀🚀  Our method does **not require pre-alignment** to standard face poses and has better consistency.
The results will be saved in the `results` folder.


🧑🏻 Video Face Restoration
**It is highly recommended to use ffmpeg to enhance the encoded video online, which can maintain the restored quality as much as possible.**
```
# Just Run This 
# Input Video Width and Height == 512
mkdir exp
python inference.py --input_video=assets/inputdemovideo.mp4 --output_video=exp/output_demo.mp4
```

### 🚀🚀🚀Awesome Video Face Restoration Method Can be Found There!🚀🚀🚀

[Awesome Video Face Restoration](https://github.com/kepengxu/Awesome-Video-Face-Restoration/tree/main)

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




## Citation

   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
@article{xu2024beyond,
  title={Beyond Alignment: Blind Video Face Restoration via Parsing-Guided Temporal-Coherent Transformer},
  author={Xu, Kepeng and Xu, Li and He, Gang and Yu, Wenxin and Li, Yunsong},
  journal={IJCAI 2024},
  year={2024}
}
   ```


## License and Acknowledgement

This project is open sourced under (https://github.com/kepengxu/PGTFormer/blob/main/LICENSE). Redistribution and use should follow this license.
The code framework is mainly modified from CodeFormer.
The page is modified from KEEP


## Contact

If you have any question, please feel free to contact us via `kepengxu11@gmail.com`.
