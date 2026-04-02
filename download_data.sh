#! /bin/bash

#python
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id='lmms-lab/LLaVA-Video-178K', repo_type='dataset', subfolder='30_60_s_nextqa', filename='30_60_s_nextqa_mc_qa_processed.json')
hf_hub_download(repo_id='lmms-lab/LLaVA-Video-178K', repo_type='dataset', subfolder='30_60_s_nextqa', filename='30_60_s_nextqa_oe_qa_processed.json')
hf_hub_download(repo_id='lmms-lab/LLaVA-Video-178K', repo_type='dataset', subfolder='30_60_s_nextqa', filename='30_60_s_nextqa_videos_1.tar.gz')
hf_hub_download(repo_id='lmms-lab/LLaVA-Video-178K', repo_type='dataset', subfolder='30_60_s_nextqa', filename='30_60_s_nextqa_videos_2.tar.gz')


#cmd
tar -xvzf hub/datasets--lmms-lab--LLaVA-Video-178K/snapshots/6d8c562dc26d70042a0d9704d1cae58c94b89098/30_60_s_nextqa/30_60_s_nextqa_videos_1.tar.gz