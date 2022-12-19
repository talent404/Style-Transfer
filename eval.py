from datasets import Dataset, load_dataset
import json
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import torch
from transformers import AdamW
from accelerate import Accelerator
import transformers
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer 
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
import nltk
from tqdm.auto import tqdm
import torch
import numpy as np
import evaluate

metric_bleu = evaluate.load('bleu')
metric_bleurt = evaluate.load('bleurt')

def preprocess_function(examples):
    # print(examples)
    inputs = [ex for ex in examples["informal"]]
    targets = [ex for ex in examples["formal"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs

def preprocess_function_eval(examples):
    # print(examples)
    inputs = [ex for ex in examples["informal"]]
    targets = [ex for ex in examples["formal.ref0"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    for i in range(4):
        model_inputs[f'formal{i}'] = examples[f'formal.ref{i}']
    return model_inputs

def collate(batch):
    keys = list(batch[0].keys())
    cols = {key:[] for key in keys}
    temp = list(map(lambda x: {'input_ids':x['input_ids'], 'labels':x['labels'], 'attention_mask':x['attention_mask']}, batch))
    for i in batch:
        for key in i:
            cols[key].append(i[key])
    out = data_collator(temp)
    for i in range(4):
        out[f'formal{i}'] = cols[f'formal{i}']
    return out

def postprocess(predictions, labels):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    return decoded_preds, decoded_labels


path = '/home/hv2237/GYAFC_Corpus/Entertainment_Music'
data = {}
for split in ['train']:
    data[split] = []
    for f, i in zip(open(f'{path}/{split}/formal').readlines(),open(f'{path}/{split}/informal').readlines()):
        data[split].append({'formal':f[:-1], 'informal':i[:-1]})
        
for split in ['tune', 'test']:
    data[split] = []
    refs = [open(f'{path}/{split}/formal.ref{i}').readlines() for i in range(4)]
    inp = open(f'{path}/{split}/informal').readlines()
    for f in range(len(inp)):
        temp = {}
        temp['informal'] = inp[f][:-1]
        for i in range(4):
            temp[f'formal.ref{i}'] = refs[i][f][:-1]
        data[split].append(temp)
        
for split in ['train', 'tune', 'test']:
    json.dump(data[split], open(f'{split}.json','w'))

train_dataset = load_dataset('json',data_files= {
    'train':'train.json'
})
eval_dataset = load_dataset('json',data_files={
    'valid':'tune.json',
    'test':'test.json'
})


tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')

tokenized_datasets = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset["train"].column_names,
)
tokenized_datasets_eval = eval_dataset.map(
    preprocess_function_eval,
    batched=True,
    remove_columns=eval_dataset["valid"].column_names,
)

model = AutoModelForSeq2SeqLM.from_pretrained('bart-finetuned/best-model').to('cuda:0')

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

tokenized_datasets.set_format("torch")

test_dataloader = DataLoader(
    tokenized_datasets_eval["test"],collate_fn=lambda x: collate(x) ,  batch_size=32
)




predictions, references = [], []
test_dataloader = accelerator.prepare(test_dataloader)

model.eval()
for batch in tqdm(test_dataloader):
    with torch.no_grad():
        generated_tokens = accelerator.unwrap_model(model).generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=128,
        )
    labels = batch["labels"]

    # Necessary to pad predictions and labels for being gathered
    generated_tokens = accelerator.pad_across_processes(
        generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
    )
    labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

    predictions_gathered = accelerator.gather(generated_tokens)
    labels_gathered = accelerator.gather(labels)

    decoded_preds, decoded_labels = postprocess(predictions_gathered, labels_gathered)
    predictions += decoded_preds
    transform = lambda x: [i for i in x]
    references += list(map(list, zip(*map(transform, [batch[f'formal{i}'] for i in range(4)]))))
 
    inps = clf_tok(decoded_preds , padding=True,truncation=True,max_length=30, return_tensors='pt').to(device)
    clf.eval()
    clf.to(device)
    clf_out = clf(**inps)
    clf_probs = torch.nn.functional.softmax(clf_out['logits'], dim=-1)
    clf_score =  clf_probs[:,1] - clf_probs[:,0]
    
    for i in range(1,4):
        bleurt.add_batch(predictions=decoded_preds, references=[i for i in batch[f'formal{i}']])

results_bleu = metric_bleu.compute(predictions=predictions, references=references)
results_bleurt = bleurt.compute()

print(f"BLEU score: {results_bleu['bleu']:.2f}, BLEURT score: {np.mean(results_bleurt['scores'])}, Classifier Score: {clf_score.mean()} ")
