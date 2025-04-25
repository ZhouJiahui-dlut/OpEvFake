# OpEvFake
Official repository for "Unveiling Opinion Evolution via Prompting and Diffusion for Short Video Fake News Detection", ACL Findings 2024.
https://aclanthology.org/2024.findings-acl.642/

## Environment:
Python 3.9

PyTorch 2.2.2

CUDA 11.8

## Data Processing:
For keyframes, video motion and audio, we used pre-extracted features from FakeSV(https://github.com/ICTMCG/FakeSV). Due to partial data loss, we pre-filtered the video IDs provided by FakeSV and saved them in the 'data/temporal_new/'.
For title&transcript, comments, user and implicit opinion, we pre-extracted the features and placed them in the 'data/' directory. You can also choose to extract features in the same way as in 'dataloader.py' from FakeSV(https://github.com/ICTMCG/FakeSV/blob/main/code/utils/dataloader.py).

## Run the Code
1. Download 'data.json' from FakeSV(https://github.com/ICTMCG/FakeSV/blob/main/dataset/data.json) and place it in the 'data/' directory.
2. As described in section 3.3 of the paper, use LLMs to generate an implicit opinion representation.
3. Use the '.txt' files in the 'data/temporal_new/' to screen the effective keyframes, video motion and audio features from FakeSV. Then, use the '.py' files in the 'fakesv_data_extract/' to extract features for text(title&transcript, comments), user, and implicit opinion separately. The feature for each modality is saved as three '.pkl' files for the training, validation, and test sets. Taking audio as an example, please save the filtered audio feature files as 'audio_train.pkl', 'audio_val.pkl', and 'audio_test.pkl' in the 'data/' directory.
4. Command as follows.
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
