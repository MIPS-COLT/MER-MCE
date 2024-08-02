# Code of MIPS at SemEval-2024 Task 3  
Official Implementation of [MIPS at SemEval-2024 Task 3: Multimodal Emotion-Cause Pair Extraction in Conversations with Multimodal Language Models](https://arxiv.org/abs/2404.00511)
## Introduction  

This paper presents a winning submission to Subtask 2 of SemEval 2024 Task 3, concentrating on multimodal emotion cause analysis in conversations. We propose a two-step architecture for multimodal emotion recognition and multimodal emotion cause extraction, our study delves into the efficacy of integrating supplementary modalities alongside textual data to bolster comprehension of emotions and discern their underlying triggers. Through experimental evaluation, our team elucidates the advantages conferred by the inclusion of diverse modalities, thereby amplifying the models' capacity for emotion understanding and causality inference. Ultimately, our submission for Subtask 2 of SemEval-2024 Task 3 attains a weighted F1 score of 0.3435, ranking third.

## Pipeline
![pipeline](./images/pipeline_prompt.jpg)

## Setup
### Prepare the code and the environment

```
git clone https://github.com/MIPS-COLT/MER-MCE.git
cd MER-MCE
conda env create -f environment.yaml
conda activate minigptv
```  

### Prepare the pretrained LLM weights
Download the Llama-2-7b-chat-hf model from Huggingface to "MER-MCE/checkpoints/"  
```
https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
```

### Prepare the pretrained model checkpoints
Download the pre-trained emotion cause analysis model on the ECAC dataset and place it in the folder named "MER-MCE/checkpoints/save_checkpoint".
```
Google Drive:
https://drive.google.com/drive/folders/1N72VyvtxdPouAPnT5fjntjXOXLBav-wi?usp=sharing

Baidu Netdisk:
https://pan.baidu.com/s/1TBOMpcplTqrjqokGHrXfZg 
Extraction Code：9jet 
```

## Run
Run the following code to extract emotional cause:  

```
torchrun  --nproc_per_node 1 eval_ECAC_cause.py --cfg-path eval_configs/minigptv2_eval_ECAC_emotion.yaml

python submit_emotion-cause_pair.py
```
Save the final submission result of Subtask 2 as "results/submit_all_cause_ck6_wd5_now-n_w-e.json" (w-avg.F1=0.3435)



If you're using MER-MCE in your research or applications, please cite using this BibTeX:
```bibtex
@inproceedings{cheng-etal-2024-mips,
    title = "{MIPS} at {S}em{E}val-2024 Task 3: Multimodal Emotion-Cause Pair Extraction in Conversations with Multimodal Language Models",
    author = "Cheng, Zebang  and
      Niu, Fuqiang  and
      Lin, Yuxiang  and
      Cheng, Zhi-qi  and
      Peng, Xiaojiang  and
      Zhang, Bowen",
    editor = {Ojha, Atul Kr.  and
      Do{\u{g}}ru{\"o}z, A. Seza  and
      Tayyar Madabushi, Harish  and
      Da San Martino, Giovanni  and
      Rosenthal, Sara  and
      Ros{\'a}, Aiala},
    booktitle = "Proceedings of the 18th International Workshop on Semantic Evaluation (SemEval-2024)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.semeval-1.97",
    doi = "10.18653/v1/2024.semeval-1.97",
    pages = "667--674",
    abstract = "This paper presents our winning submission to Subtask 2 of SemEval 2024 Task 3 on multimodal emotion cause analysis in conversations. We propose a novel Multimodal Emotion Recognition and Multimodal Emotion Cause Extraction (MER-MCE) framework that integrates text, audio, and visual modalities using specialized emotion encoders. Our approach sets itself apart from top-performing teams by leveraging modality-specific features for enhanced emotion understanding and causality inference. Experimental evaluation demonstrates the advantages of our multimodal approach, with our submission achieving a competitive weighted F1 score of 0.3435, ranking third with a margin of only 0.0339 behind the 1st team and 0.0025 behind the 2nd team.",
}
