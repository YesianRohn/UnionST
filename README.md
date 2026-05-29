# UnionST: A Strong Synthetic Engine for Scene Text Recognition

<a href='https://arxiv.org/abs/2602.06450'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://huggingface.co/Yesianrohn/UnionST-Models'><img src='https://img.shields.io/badge/Ckpt-Huggingface-yellow'></a> 
<a href='https://huggingface.co/datasets/Yesianrohn/UnionST'><img src='https://img.shields.io/badge/Data-Huggingface-purple'></a>

Official data synthesis code of the paper *"What’s Wrong with Synthetic Data for Scene Text Recognition? A Strong Synthetic Engine with Diverse Simulations and Self-Evolution"*.

## Introduction
Scene Text Recognition (STR) relies critically on large-scale, high-quality training data. While synthetic data provides a cost-effective alternative to manually annotated real data, existing rendering-based synthetic datasets suffer from **insufficient diversity** (corpus/font/layout) and a large domain gap with real-world text. 

### Key Advantages
- 🎯 **100% Label Correctness**: Rendering-based paradigm ensures accurate labels (unlike generative models with aesthetic but error-prone outputs).
- ⚡ **Cost-Efficiency**: CPU-based generation costs only 1/20 of diffusion-based methods and 1/10,000 of closed-source alternatives.
- 🚀 **Strong Performance**: UnionST-S (5M samples) outperforms 36M-scale traditional synthetic datasets on challenging STR benchmarks.


## Dataset

UnionST-S, UnionST-P, and UnionST-R datasets (each containing 5M samples) can be downloaded from [Huggingface](https://huggingface.co/datasets/Yesianrohn/UnionST). We use the lmdb file format adopted by the mainstream STR protocol. In addition, we have  summarized the other STR synthetic datasets compared in the paper, which are available [here](https://huggingface.co/datasets/Yesianrohn/STR-Synth).

## Installation

1. **Clone the UnionST Repository:**
   
    ```bash
    git clone https://github.com/YesianRohn/UnionST.git
    cd UnionST
   ```

2. **Create a New Environment for UnionST:**
   ```bash
   conda create -n unionst python=3.8
   conda activate unionst
   ```

3. **Install Required Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Resource Preparation

#### 1. Background Preparation

We directly use the 8,000 background images provided by [SynthText]((https://github.com/ankush-me/SynthText)), and you can download them from the official repository.

#### 2. Font Preparation

- You can use your own font files. Due to copyright restrictions, we cannot provide the font library we used. As an alternative, you can use [Google Fonts](https://github.com/google/fonts/archive/main.zip) as a base.
- Font filtering:
  ```bash
  python tools/filter_font.py
  ```
- Extract renderable charsets:
  ```bash
  python tools/extract_font_charset.py -w 4 fonts/
  ```

For additional operations and extensions, please refer to [SynthTIGER](https://github.com/clovaai/synthtiger).

If you do not wish to perform the above preparations, we provide some simple files in the corresponding locations, so you can directly proceed with the following steps.

## Usage

#### Visualization Generation

```bash
python main.py -w 16 -v config/template.py UnionST config/config.yaml -o results/vis -c 100
```

#### Dataset Generation (LMDB Format)

```bash
python main.py -w 16 -v config/template.py UnionST config/config.yaml -o results/lmdb -c 100 --lmdb
```

## Dataset

UnionST-S, UnionST-P, and UnionST-R datasets (each containing 5M samples) can be downloaded from [Huggingface](h) and [Modelscope](m).

## Training Model
[OpenOCR](https://github.com/Topdu/OpenOCR)

```bash
cd OpenOCR
torchrun  --nproc_per_node=8 tools/train_rec.py --c configs/rec/nrtr/svtrv2_nrtr_unionst.yml
```

Some of our trained models can be found at [Huggingface](https://huggingface.co/Yesianrohn/UnionST-Models).

## Citation
   ```bash
@inproceedings{ye2026wrong,
  title={What's Wrong with Synthetic Data for Scene Text Recognition? A Strong Synthetic Engine with Diverse Simulations and Self-Evolution},
  author={Ye, Xingsong and Du, Yongkun and Zhang, JiaXin and Li, Chen and LYU, Jing and Chen, Zhineng},
  booktitle={CVPR},
  year={2026}
}
   ```

## License
   ```bash
"""
UnionST
Copyright (c) 2025-present YesianRohn
Based on SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT License
"""

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
   ```


## Acknowledgements
- We thank the [SynthText](https://github.com/ankush-me/SynthText), [SynthTIGER](https://github.com/clovaai/synthtiger), [SVTRv2](https://github.com/Topdu/OpenOCR/blob/main/docs/svtrv2.md) and [Union14M](https://github.com/Mountchicken/Union14M) for their open-source code/datasets.
- Special thanks also go to the training framework: [OpenOCR](https://github.com/Topdu/OpenOCR).


