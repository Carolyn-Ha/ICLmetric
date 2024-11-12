# Setup
## Installation

**Installation for local development via conda**

```
git clone https://github.com/Romainpkq/revisit_demon_selection_in_ICL.git
conda install python=3.8
conda install -c pytorch -c nvidia faiss-gpu=1.7.2
pip install -e .
pip install sentencepiece # For Vicuna tokenizer
```

Reference
- [link](https://github.com/facebookresearch/faiss/blob/b77061ff5eb2d5dc3b1fc25b240578c2d686a646/INSTALL.md)

# Run

## Examples

For single GPU server, use following scripts which is described in scripts/run.sh
```
CUDA_VISIBLE_DEVICES=$DEVICES accelerate launch exp/run.py

# calculate the accuracy
python prediction.py
```

Reference
- [examples](https://github.com/Shark-NLP/OpenICL/tree/main/examples)

# TODO

- Update python/bash scripts for experiments
- Update src according to the experiment factors!