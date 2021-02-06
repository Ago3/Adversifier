from transformers import BertTokenizer, BertForSequenceClassification
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
from argparse import ArgumentParser
from utils import preprocess_tweet


LABELS = ['neither', 'sexism', 'racism', 'both']


def batcher(lines, batch_size=100):
    batch = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(batch) >= batch_size:
            yield batch
            batch = []
        batch.append(line)
    if batch:
        yield batch


def _string_to_label_(label):
    return torch.tensor([int(c) for c in ('0' + '{0:b}'.format(LABELS.index(label.strip())))[-2:]])


class KennedyModel():
    def __init__(self, racism_model, sexism_model, batch_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.racism_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).eval()
        state_dict = torch.load(racism_model, map_location='cpu')
        state_dict['bert.embeddings.position_ids'] = self.racism_model.state_dict()['bert.embeddings.position_ids']
        self.racism_model.load_state_dict(state_dict)
        self.racism_model.to(self.device)

        self.sexism_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).eval()
        state_dict = torch.load(sexism_model, map_location='cpu')
        state_dict['bert.embeddings.position_ids'] = self.sexism_model.state_dict()['bert.embeddings.position_ids']
        self.sexism_model.load_state_dict(state_dict)
        self.sexism_model.to(self.device)

    def forward(self, input_args):
        input_lines = input_args[0]  # this model only takes the posts as input
        input_lines = [preprocess_tweet(tweet) for tweet in input_lines]
        predictions = None
        for batch_lines in tqdm(batcher(input_lines, batch_size=self.batch_size)):
            batch = self.tokenizer.batch_encode_plus(batch_lines, return_tensors='pt', padding='longest', max_length=512)
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.no_grad():
                racism_predictions = torch.tensor(self.racism_model(**batch)[0].argmax(-1).tolist())
                sexism_predictions = torch.tensor(self.sexism_model(**batch)[0].argmax(-1).tolist())

            current_predictions = racism_predictions | sexism_predictions
            if predictions is None:
                predictions = current_predictions
            else:
                predictions = torch.cat([predictions, current_predictions], dim=0)

        return predictions
