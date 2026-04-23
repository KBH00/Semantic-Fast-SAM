## Semantic-Fast-SAM
https://arxiv.org/abs/2604.20169
https://ieeexplore.ieee.org/document/11249315
## Installation

To install the required dependencies, run the following command:

```bash
git clone https://github.com/KBH00/Semantic-Fast-SAM.git
```
```
cd Semantic-Fast-SAM
```
```
conda env create -f environment.yaml
```
```
conda activate sfs
```
```
python -m spacy download en_core_web_sm
```

You need to download Fast-SAM model checkpoint in [here](https://drive.google.com/file/d/1l7l1VJmpD1nOsgiTXucTtYOpu3nE-rjh/view?usp=sharing)\
Put FastSAM.pt in weights directory, mabye you should change the file name FastSAM-x.pt to FastSAM.pt


## Inference
```
python scripts/main_ssa_engine.py --data_dir=data/<The name of your image> --out_dir=output --world_size=<choose GPU number>
```
Or, you can just run main_ssa_engine.py

## Examples
![Description of Image](./pngs/cat.png)

![Description of Image](./pngs/dogs.png)

## References

Fast Segment Anything
https://github.com/CASIA-IVA-Lab/FastSAM

Semantic Segment Anything
https://github.com/fudan-zvg/Semantic-Segment-Anything

