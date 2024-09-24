# OpEvFake
Official repository for "Unveiling Opinion Evolution via Prompting and Diffusion for Short Video Fake News Detection", ACL Findings 2024.
https://aclanthology.org/2024.findings-acl.642/

## Dataset:
FakeSV: https://github.com/ICTMCG/FakeSV

## Environment:
Python 3.9

PyTorch 2.2.2

CUDA 11.8

## Run the Code
1. As described in section 3.3 of the paper, use LLMs to generate an implicit opinion representation.
2. Place the FakeSV features and implicit opinion representation features in the 'data/' directory. Then, create the `dataloader_fakesv.py` file containing the `get_dataloader` function.
3. Command as follows.
```
python main.py
```

## Citation:
@inproceedings{DBLP:conf/acl/ZongZLL0024,
  author       = {Linlin Zong and
                  Jiahui Zhou and
                  Wenmin Lin and
                  Xinyue Liu and
                  Xianchao Zhang and
                  Bo Xu},
                  
  editor       = {Lun{-}Wei Ku and
                  Andre Martins and
                  Vivek Srikumar},
                  
  title        = {Unveiling Opinion Evolution via Prompting and Diffusion for Short
                  Video Fake News Detection},
                  
  booktitle    = {Findings of the Association for Computational Linguistics, {ACL} 2024,
                  Bangkok, Thailand and virtual meeting, August 11-16, 2024},
                  
  pages        = {10817--10826},
  
  publisher    = {Association for Computational Linguistics},
  
  year         = {2024},
  
  url          = {https://aclanthology.org/2024.findings-acl.642},
  
  timestamp    = {Tue, 27 Aug 2024 17:38:11 +0200},
  
  biburl       = {https://dblp.org/rec/conf/acl/ZongZLL0024.bib},
  
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
