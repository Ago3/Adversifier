import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score
from utils import preprocess_tweet
import torch.nn as nn
import numpy as np


class BertModel(nn.Module):
    def __init__(self, num_labels, tokenizer=None):
        super(BertModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained("bert-base-uncased")

    def run_inference(self, input_args):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert.to(device)

        input_lines = input_args[0]
        input_lines = [preprocess_tweet(tweet, use_hashtags=True) for tweet in input_lines]

        all_preds = []
        self.bert.eval()
        with torch.no_grad():
            for input_batch in self.__get_batch__(input_lines):
                encodings = self.tokenizer(
                    input_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                ).to(device)

                outputs = self.bert(**encodings)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())

        return torch.cat(all_preds, dim=0)

    def __get_batch__(self, full_input, batch_size=32):
        for i in range(0, len(full_input), batch_size):
            yield full_input[i:min(i + batch_size, len(full_input))]


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def define_model(num_labels=2):
    model = BertModel(num_labels=num_labels)
    return model


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='micro')
    return {"f1": f1}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def finetune_bert_model(train_data, val_data, dataset_name, model=None, epochs=3, batch_size=16, seed=74361):
    set_seed(seed)

    train_texts = [preprocess_tweet(tweet, use_hashtags=True) for tweet in train_data[0]]
    train_labels = train_data[1]
    val_texts = [preprocess_tweet(tweet, use_hashtags=True) for tweet in val_data[0]]
    val_labels = val_data[1]

    if model is None:
        model = define_model(num_labels=len(set(train_labels)))

    train_dataset = TextDataset(train_texts, train_labels, model.tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, model.tokenizer)

    training_args = TrainingArguments(
        output_dir=f"./CACHE/bert_results_{dataset_name}",
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=1,
        logging_dir="./CACHE/logs",
        learning_rate=2e-5,
        logging_steps=50,
        evaluation_strategy="steps",
        save_steps=200,
        eval_steps=200
    )

    trainer = Trainer(
        model=model.bert,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    checkpoint_path = None
    best_model_path = f"./models/best_bert_model_{dataset_name}"
    best_f1 = -np.inf

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        trainer.args.num_train_epochs = 1
        trainer.train(resume_from_checkpoint=checkpoint_path)
        metrics = trainer.evaluate()
        checkpoint_path = f"./CACHE/bert_results_{dataset_name}/checkpoint-last"
        print(f"Epoch {epoch + 1} — F1: {metrics['eval_f1']:.4f}")
        if metrics['eval_f1'] > best_f1:
            best_f1 = metrics['eval_f1']
            trainer.save_model(best_model_path)
            print(f"Best model updated at epoch {epoch + 1}")

    best_model = BertForSequenceClassification.from_pretrained(best_model_path)
    model.bert = best_model
    return model
