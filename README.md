# Style-Transfer
The task aims to generate formal version of a sentence given an informal input. 

## Dataset
The Grammarly's Yahoo Answers Formality Corpus (GYAFC) is not publicly available. It was created using the Yahoo Answers corpus: L6 - Yahoo! Answers Comprehensive Questions and Answers version 1.0 . This Yahoo Answers corpus can be requested free of charge for research purposes.
Access to our GYAFC dataset will require users to first gain access to this Yahoo Answers corpus.

Follow the instructions present at https://github.com/raosudha89/GYAFC-corpus to obtain the annotations.

## Training 
### Baselines
To train a sequence to sequence huggingface model on the given domain of the data set (E&R or F&M), <br><br>
`python train.py --data <path-to-data-dir> --model <huggingface-model>`

### RL based reward
To train a sequence to sequence huggingface model on the given domain of the data set (E&R or F&M), using the following reward functions
* BLEU
* BLEURT
* Style Classifier score <br><br>
`python train.py --data <path-to-data-dir> --model <huggingface-model> --bleu --bleurt --classifier --classifier_model <huggingface-model>`

The style classifier notebook present in the Notebooks directory can be used to train a custom distilbert based formality classifier.

### Inverse Reinforcement learning 
When using multiple RL based rewards using the above script, equal weights are applied to each reward. To learn the optimal weights for each reward weights, Inverse reinforcement leraning is used. The below script finetunes the given huggingface model, using all the above mentioned rewards.
`python train_irl.py --dataset <path-to-data-dir> --model <huggingface-model> --classifier_model <huggingface-model>`

## Evaluation
The model can be evaluated on all the reward metrics it was trained on using,<br><br>
`python eval.py --dataset <path-to-data-dir> --model <huggingface-model> --classifier_model <huggingface-model>`

## Huggingface Hub
The best performing model, which is a bart-base model finetuned with all the reward functions, can be used at https://huggingface.co/talent404/formal-generation
