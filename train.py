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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Huggingface model to be used", required=True)
parser.add_argument("--data", help="Path to E&R or F&M datasets", required=True)
parser.add_argument("--bleu", help="Finetune with BLEU reward", default=False, action='store_true')
parser.add_argument("--bleurt", help="Finetune with BLEURT reward", default=False, action='store_true')
parser.add_argument("--classifier", help="Finetune with style classifier reward", default=False, action='store_true')
parser.add_argument("--classifier_model", help="path to pretrained classifier model")


args = parser.parse_args()


max_length = 30
bleurt = evaluate.load('bleurt', module_type="metric", checkpoint='bleurt-tiny-128')
metric_bleu = evaluate.load('bleu')
clf = AutoModelForSequenceClassification.from_pretrained(args.classifier_model, num_labels=2)
clf_tok = AutoTokenizer.from_pretrained(args.classifier_model)
smooth = SmoothingFunction()
nltk.download('punkt')

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


def rl_loss(probs, reward):
    # Computing the policy gradient
    reward = reward.unsqueeze(1)
    sampled_logprobs = torch.log(probs)
    r_loss = -sampled_logprobs*reward
    r_loss = r_loss.mean()
    return r_loss

def bleurt_loss(batch):
    labels = batch['labels']
    
    # Sampling from policy
    sampled_outputs = model.generate(
          input_ids=batch['input_ids'],
          attention_mask=batch['attention_mask'],
          output_scores=True,
          num_beams=1,
          do_sample=True,
          return_dict_in_generate=True,
          max_length=30,  
          top_k=tokenizer.vocab_size
        )
    
    # Greedy decoding for baseline
    greedy_outputs = model.generate(
          input_ids=batch['input_ids'],
          attention_mask=batch['attention_mask'],
          output_scores=True,
          do_sample=False,
          num_beams=1,
          max_length=30,
          return_dict_in_generate=True,
        )
    
    # Obtaining the probabilities of sampled tokens
    out = [i for i in sampled_outputs.scores[0]]
    for i in range(1,len(sampled_outputs.scores)):
        for j in range(len(out)):
            out[j] = torch.vstack((out[j],sampled_outputs.scores[i][j]))
    scores = torch.vstack(out).reshape(sampled_outputs.scores[0].shape[0], len(sampled_outputs.scores),-1)
    scores = torch.nn.functional.softmax(scores,dim=-1)
    sample_idx, sampled_probs = torch.zeros(scores.shape[0], scores.shape[1]).to(accelerator.device), torch.zeros(scores.shape[0], scores.shape[1]).to(accelerator.device)
    for ind, i in enumerate(scores):
        temp = torch.multinomial(i, 1)
        sampled_probs[ind] = i.gather(1, temp.type(torch.int64)).squeeze(1)
        sample_idx[ind] = temp.squeeze(1)
    
    sampled_texts = tokenizer.batch_decode(sampled_outputs.sequences, skip_special_tokens=True)
    greedy_texts = tokenizer.batch_decode(greedy_outputs.sequences, skip_special_tokens=True)
    labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
    label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)


    sample_bleurt = bleurt.compute(references=label_texts, predictions=sampled_texts)['scores']
    greedy_bleurt = bleurt.compute(references=label_texts, predictions=greedy_texts)['scores']
    sample_bleurt = torch.FloatTensor(sample_bleurt).to(accelerator.device)
    greedy_bleurt = torch.FloatTensor(greedy_bleurt).to(accelerator.device)
    
    reward = (greedy_bleurt - sample_bleurt)
    
    return rl_loss(sampled_probs, reward)



def classifier_loss(batch):
    sampled_outputs = model.generate(
          input_ids=batch['input_ids'],
          attention_mask=batch['attention_mask'],
          output_scores=True,
          num_beams=1,
          do_sample=True,
          return_dict_in_generate=True,
          max_length=30,  
          top_k=tokenizer.vocab_size
        )
    out = [i for i in sampled_outputs.scores[0]]
    for i in range(1,len(sampled_outputs.scores)):
        for j in range(len(out)):
            out[j] = torch.vstack((out[j],sampled_outputs.scores[i][j]))
    scores = torch.vstack(out).reshape(sampled_outputs.scores[0].shape[0], len(sampled_outputs.scores),-1)
    scores = torch.nn.functional.softmax(scores,dim=-1)

    sample_idx, sampled_probs = torch.zeros(scores.shape[0], scores.shape[1]).to(accelerator.device), torch.zeros(scores.shape[0], scores.shape[1]).to(accelerator.device)
    for ind, i in enumerate(scores):
        temp = torch.multinomial(i, 1)
        sampled_probs[ind] = i.gather(1, temp.type(torch.int64)).squeeze(1)
        sample_idx[ind] = temp.squeeze(1)
    sampled_texts = tokenizer.batch_decode(sampled_outputs.sequences, skip_special_tokens=True)
    
    # Obtaining style classifier scores 
    
    inps = clf_tok(sampled_texts , padding=True,truncation=True,max_length=30, return_tensors='pt').to(device)
    
    clf.to(device)
    clf.eval()
    clf_out = clf(**inps)
    clf_probs = torch.nn.functional.softmax(clf_out['logits'], dim=-1)
    clf_reward =  clf_probs[:,1] - clf_probs[:,0] 
    
    return rl_loss(sampled_probs, clf_reward)

def bleu_loss(batch):
    labels = batch['labels']
    sampled_outputs = model.generate(
          input_ids=batch['input_ids'],
          attention_mask=batch['attention_mask'],
          output_scores=True,
          num_beams=1,
          do_sample=True,
          return_dict_in_generate=True,
          max_length=30,  
          top_k=tokenizer.vocab_size
        )
    greedy_outputs = model.generate(
          input_ids=batch['input_ids'],
          attention_mask=batch['attention_mask'],
          output_scores=True,
          do_sample=False,
          num_beams=1,
          max_length=30,
          return_dict_in_generate=True,
        )
    
    out = [i for i in sampled_outputs.scores[0]]
    for i in range(1,len(sampled_outputs.scores)):
        for j in range(len(out)):
            out[j] = torch.vstack((out[j],sampled_outputs.scores[i][j]))
    scores = torch.vstack(out).reshape(sampled_outputs.scores[0].shape[0], len(sampled_outputs.scores),-1)
    scores = torch.nn.functional.softmax(scores,dim=-1)
    sample_idx, sampled_probs = torch.zeros(scores.shape[0], scores.shape[1]).to(accelerator.device), torch.zeros(scores.shape[0], scores.shape[1]).to(accelerator.device)
    for ind, i in enumerate(scores):
        temp = torch.multinomial(i, 1)
        sampled_probs[ind] = i.gather(1, temp.type(torch.int64)).squeeze(1)
        sample_idx[ind] = temp.squeeze(1)
    
    sampled_texts = tokenizer.batch_decode(sampled_outputs.sequences, skip_special_tokens=True)
    greedy_texts = tokenizer.batch_decode(greedy_outputs.sequences, skip_special_tokens=True)
    labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
    label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)


    sample_bleu, greedy_bleu = [], []
    i = 0
    for ref, pred in zip(label_texts, sampled_texts):
        if len(pred)==0:
            sample_bleu.append(0)
            continue
        sample_bleu.append(sentence_bleu([ref], pred, smoothing_function=smooth.method1))
        i+=1
    i = 0
    for ref, pred in zip(label_texts, greedy_texts):
        if len(pred)==0:
            greedy_bleu.append(0)
            continue
        greedy_bleu.append(sentence_bleu([ref], pred, smoothing_function=smooth.method1))
        i+=1

    sample_bleu = torch.FloatTensor(sample_bleu).to(accelerator.device)
    greedy_bleu = torch.FloatTensor(greedy_bleu).to(accelerator.device)
    
    reward = (greedy_bleu - sample_bleu)
    
    return rl_loss(sampled_probs, reward)
    

path = f'{args.data}'
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


tokenizer = AutoTokenizer.from_pretrained(args.model)

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

model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

tokenized_datasets.set_format("torch")
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    collate_fn=data_collator,
    batch_size=32,
)
eval_dataloader = DataLoader(
    tokenized_datasets_eval["valid"],collate_fn=lambda x: collate(x) ,  batch_size=32
)

test_dataloader = DataLoader(
    tokenized_datasets_eval["test"],collate_fn=lambda x: collate(x) ,  batch_size=32
)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

optimizer = torch.optim.Adam(model.parameters(),
                         betas=(0.9, 0.98), eps=1e-09, lr=3e-05, weight_decay=0.01)

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

num_train_epochs = 10
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = transformers.get_polynomial_decay_schedule_with_warmup(optimizer, 500, 20000)



progress_bar = tqdm(range(num_training_steps))
steps = 0
best_bleu = 0.0
for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        
        
        
        if steps>200:
            if args.bleu:
                loss += bleu_loss(batch)
            if args.bleurt:
                loss += bleurt_loss(bacth)
            if args.classifier:
                loss += classifier_loss(batch)
        
        steps+=1
        
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        # Evaluation
        if steps > 0 and steps%200 == 0:
            model.eval()
            predictions, references = [], []
            for batch in tqdm(eval_dataloader):
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


            results_bleu = metric_bleu.compute(predictions=predictions, references=references)

            print(f"epoch {epoch}, BLEU score: {results_bleu['bleu']:.2f}")

            # Save and upload
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            
            if results_bleu['bleu']>best_bleu:
                unwrapped_model.save_pretrained(f'{args.model}-finetuned-fm/best-model', save_function=accelerator.save)
                best_bleu = results_bleu['bleu']
            unwrapped_model.save_pretrained(f'{args.model}-finetuned-fm', save_function=accelerator.save)
