# OpEvFake
The implementation of the paper Unveiling Opinion Evolution via Prompting and Diffusion for Short Video Fake News Detection.

## Dataset:
FakeSV: https://github.com/ICTMCG/FakeSV

## Environment:
Python 3.9

Pytorch 2.2.2

CUDA 11.8

## Run the Code
1. Put the downloaded data in 'data/'. Create the `dataloader_fakesv.py` file that contains the get_dataloader function.
2. Command as follows.
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
