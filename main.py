#%%

#%%
from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering

tokenizer = RobertaTokenizerFast.from_pretrained('nur-dev/roberta-kaz-large')
model = RobertaForQuestionAnswering.from_pretrained('nur-dev/roberta-kaz-large')
#%%
from datasets import load_dataset, concatenate_datasets

# valid_ds = load_dataset("issai/kazqad", "kazqad", split="validation")
test_ds = load_dataset("issai/kazqad", "kazqad", split="test")
test_ds = test_ds.select(range(1000))
# dataset = load_dataset("Kyrmasch/sKQuAD", "kazqad", split="train")
# dataset2 = load_dataset("issai/kazqad", "nq-translate-kk", split="train")
# 
# dataset = concatenate_datasets([dataset1, dataset2, valid_ds])
#%%
dataset = load_dataset("Kyrmasch/sKQuAD", "default", split="train")
#%%
def tokenize_function(examples):
    return tokenizer(
        examples["question"],
        examples["context"],
        padding="max_length",
        truncation=True,
        max_length=384,
        return_overflowing_tokens=True,
        return_offsets_mapping=True
    )
#%%
def prepare_train_features(examples):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    
    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    
    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answer"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            
            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            
            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
    
    return tokenized_examples

#%%
def add_answer_start(example):
    context = example['context']
    answer_text = example['answer']
    
    # Find the position of the answer in the context
    start_idx = context.find(answer_text)
    
    if start_idx == -1:
        # Answer not found in context
        start_idx = None
    example['answers'] = {
        'text': [answer_text],
        'answer_start': [start_idx] if start_idx is not None else []
    }
    return example
#%%
dataset = dataset.map(add_answer_start)
#%%
def filter_missing_answers(example):
    return len(example['answers']['answer_start']) > 0

dataset = dataset.filter(filter_missing_answers)
#%%
def prepare_train_features(examples):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    
    # Initialize start and end positions
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        
        if len(answers["answer_start"]) == 0:
            # If no answer is found, set start and end positions to CLS index
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            
            # Find the start and end token indices
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            
            # If the answer is not fully inside the context, label as CLS index
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Move the token indices to the answer boundaries
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_position = token_start_index - 1
                
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_position = token_end_index + 1
                
                tokenized_examples["start_positions"].append(start_position)
                tokenized_examples["end_positions"].append(end_position)
    
    return tokenized_examples

#%%
tokenized_datasets = dataset.map(
    prepare_train_features,
    batched=True,
    remove_columns=dataset.column_names
)
#%%
tokenized_datasets = dataset.map(
    prepare_train_features,
    batched=True,
    remove_columns=dataset.column_names
)
#%%
tokenized_test_datasets = test_ds.map(
    prepare_train_features,
    batched=True,
    remove_columns=test_ds.column_names
)
#%%

#%%
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    start_logits, end_logits = logits
    
    start_preds = np.argmax(start_logits, axis=-1)
    end_preds = np.argmax(end_logits, axis=-1)
    
    f1 = f1_score(labels[0], start_preds, average="weighted")
    accuracy = accuracy_score(labels[0], start_preds)
    
    return {"f1": f1, "accuracy": accuracy}

#%%
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    #save_strategy="epoch",
    learning_rate=3e-06,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    adam_beta1=0.8,
    adam_beta2=0.999,
    num_train_epochs=30,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    report_to="tensorboard",
    #deepspeed="ds_config.json",
    bf16=True
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_test_datasets,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
#%%
trainer.train()
#%%
