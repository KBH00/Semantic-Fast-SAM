# Semantic-Fast-SAM: Efficient Semantic Segmenter

Official implementation of **Semantic-Fast-SAM: Efficient Semantic Segmenter**.

**Author:** Byunghyun Kim  
**arXiv:** https://arxiv.org/abs/2604.20169  
**arXiv DOI:** https://doi.org/10.48550/arXiv.2604.20169  
**IEEE Xplore:** https://ieeexplore.ieee.org/document/11249315  
**IEEE DOI:** https://doi.org/10.1109/APSIPAASC65261.2025.11249315  

## Paper

**Semantic-Fast-SAM: Efficient Semantic Segmenter**  
Byunghyun Kim

Semantic-Fast-SAM (SFS) is an efficient semantic segmentation framework that combines FastSAM mask generation with semantic labeling. It produces semantic segmentation maps with substantially lower computational cost than SAM-based semantic segmentation pipelines, while retaining the segment-anything capability for closed-set and open-vocabulary segmentation.

The paper is available as an arXiv preprint and as an IEEE APSIPA ASC proceedings paper.

- **Preprint:** arXiv:2604.20169 [cs.CV]
- **Published version:** Proceedings of the 2025 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), IEEE, Singapore, 2025, pp. 1158-1163

## Installation

Clone this repository:

```bash
git clone https://github.com/KBH00/Semantic-Fast-SAM.git
cd Semantic-Fast-SAM
```

Create and activate the conda environment:

```bash
conda env create -f environment.yaml
conda activate sfs
```

Install the required spaCy model:

```bash
python -m spacy download en_core_web_sm
```

Download the FastSAM checkpoint from the following link:

```text
https://drive.google.com/file/d/1l7l1VJmpD1nOsgiTXucTtYOpu3nE-rjh/view?usp=sharing
```

Place the checkpoint in the `weights/` directory. If the downloaded file is named `FastSAM-x.pt`, rename it to `FastSAM.pt`.

Expected path:

```text
weights/FastSAM.pt
```

## Inference

Run inference with:

```bash
python scripts/main_ssa_engine.py \
  --data_dir data/<image_or_directory_name> \
  --out_dir output \
  --world_size <number_of_gpus>
```

For example:

```bash
python scripts/main_ssa_engine.py \
  --data_dir data/example.jpg \
  --out_dir output \
  --world_size 1
```

You can also run `scripts/main_ssa_engine.py` directly after setting the required arguments in the script.

## Examples

![Semantic-Fast-SAM example: cat](./pngs/cat.png)

![Semantic-Fast-SAM example: dogs](./pngs/dogs.png)

## Citation

If you use this repository, please cite the paper.

### arXiv version

```bibtex
@article{kim2026semanticfastsam,
  title        = {Semantic-Fast-SAM: Efficient Semantic Segmenter},
  author       = {Kim, Byunghyun},
  journal      = {arXiv preprint arXiv:2604.20169},
  year         = {2026},
  doi          = {10.48550/arXiv.2604.20169},
  url          = {https://arxiv.org/abs/2604.20169}
}
```

### IEEE APSIPA ASC version

```bibtex
@inproceedings{kim2025semanticfastsam,
  title        = {Semantic-Fast-SAM: Efficient Semantic Segmenter},
  author       = {Kim, Byunghyun},
  booktitle    = {Proceedings of the 2025 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)},
  pages        = {1158--1163},
  year         = {2025},
  publisher    = {IEEE},
  address      = {Singapore},
  doi          = {10.1109/APSIPAASC65261.2025.11249315},
  url          = {https://doi.org/10.1109/APSIPAASC65261.2025.11249315}
}
```

## Paper Metadata

For better discoverability and citation-tool indexing of the arXiv version, keep the paper metadata consistent across this repository, the arXiv page, the IEEE page, and any project website:

- Title: `Semantic-Fast-SAM: Efficient Semantic Segmenter`
- Author: `Byunghyun Kim`
- arXiv ID: `2604.20169`
- arXiv DOI: `10.48550/arXiv.2604.20169`
- IEEE DOI: `10.1109/APSIPAASC65261.2025.11249315`

A dedicated project page with Google Scholar citation meta tags is recommended for stronger indexing. For example, a GitHub Pages site can expose `citation_title`, `citation_author`, `citation_publication_date`, `citation_doi`, `citation_arxiv_id`, and `citation_pdf_url` fields in the HTML header.

## Related Projects

This project builds on the following repositories:

- Fast Segment Anything: https://github.com/CASIA-IVA-Lab/FastSAM
- Semantic Segment Anything: https://github.com/fudan-zvg/Semantic-Segment-Anything

## License

Please check the license terms of this repository and the related projects before using the code, models, or checkpoints.
