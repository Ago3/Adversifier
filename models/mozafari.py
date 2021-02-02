import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset
from info import DROPOUT, NUM_CLASSES, THRESHOLD, BATCH_SIZE


class WaseemDataset(Dataset):

    def __init__(self, data):
        """
        Args:
            data (dict): Mapping from instance id to [tweet id, text, [label1, label2, ..., labeln]]
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.LABELS = ['neither', 'sexism', 'racism', 'both']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = dict()
        sample['nInst'] = idx
        sample['id'], sample['text'], _ = self.data[idx]
        sample['labels'] = torch.tensor([int(c) for c in ('0' + '{0:b}'.format(self.LABELS.index(self.data[idx][2])))[-2:]])
        return sample


class MozafariModel(nn.Module):
    def __init__(self, mozafari_model):
        super(MozafariModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.fc1 = weight_norm(nn.Linear(768, NUM_CLASSES, bias=True))
        self.dropout = nn.Dropout(p=DROPOUT)
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.sigmoid = nn.Sigmoid()
        self.load_state_dict(torch.load(mozafari_model, map_location='cpu'))

    def forward(self, sample_batched):
        tweet_emb = self.__bert_encoding__(self.bert_model, sample_batched['text'])  # [BATCH, 768]
        predictions = self.dropout(self.fc1(tweet_emb).float())  # [BATCH, NUM_CLASSES]
        # if self.training:
        #     loss = self.loss(predictions, sample_batched['labels'].float().to(self.device))  # [1]
        #     return loss
        # else:
        # confidence_scores = self.sigmoid(predictions)
        # high_confidence_predictions = (self.sigmoid(predictions) > 0.9).sum().long() + (self.sigmoid(predictions) < 0.1).sum().long()
        predictions = (self.sigmoid(predictions) >= THRESHOLD).long()
        binary_pred = predictions.max(dim=1)[0]
        # binary_labels = sample_batched['labels'].max(dim=1)[0]
        # tp, tn, fp, fn, correct = self.__get_confusion_matrix__(predictions, sample_batched['labels'])
        # labels_3c = self.__convert_to_3_classes__(sample_batched['labels'])
        # predictions_3c = self.__convert_to_3_classes__(predictions)
        # precision_3c = precision_score(labels_3c, predictions_3c, average='micro')
        # recall_3c = recall_score(labels_3c, predictions_3c, average='micro')
        # return correct.item(), tp.item(), tn.item(), fp.item(), fn.item(), predictions, labels_3c, predictions_3c, high_confidence_predictions.item(), confidence_scores, binary_pred, binary_labels
        return binary_pred

    def __bert_encoding__(self, bert, sentences):
        import tensorflow as tf
        ids = []
        for sent in sentences:
            ids.append(self.tokenizer.encode(sent, add_special_tokens=True))
        maxlen = max(len(i) for i in ids)
        x = torch.tensor([i + [self.tokenizer.pad_token_id] * (maxlen - len(i)) for i in ids]).to(self.device)
        mask = (x != self.tokenizer.pad_token_id)
        outputs = bert(x, attention_mask=mask)
        last_hidden_states = outputs[0][:, 0, :]  # Take hidden for CLS token
        return last_hidden_states

    # def __get_confusion_matrix__(self, predictions, labels):
    #     tp = ((predictions == labels.to(self.device)) * (predictions == torch.ones(labels.shape, device=self.device).long())).sum()  # [1]
    #     tn = ((predictions == labels.to(self.device)) * (predictions == torch.zeros(labels.shape, device=self.device).long())).sum()  # [1]
    #     fp = ((predictions != labels.to(self.device)) * (predictions == torch.ones(labels.shape, device=self.device).long())).sum()  # [1]
    #     fn = ((predictions != labels.to(self.device)) * (predictions == torch.zeros(labels.shape, device=self.device).long())).sum()  # [1]
    #     correct = (predictions == labels.to(self.device)).sum()  # [1]
    #     return tp, tn, fp, fn, correct

    # def __convert_to_3_classes__(self, old):
    #     classes_11 = (old.to(self.device) == torch.tensor([[1, 1]], device=self.device)).sum(dim=1)
    #     classes_10 = (old.to(self.device) == torch.tensor([[1, 0]], device=self.device)).sum(dim=1)
    #     new = torch.where(classes_11 == torch.tensor([1], device=self.device), torch.where(classes_10 == torch.tensor([2], device=self.device), torch.tensor([2], device=self.device), torch.tensor([1], device=self.device)), classes_11)
    #     return new

    def predictor(self, input_args):
        input_lines = input_args[0]  # this model only takes the posts as input
        input_lines = [preprocess_tweet(tweet) for tweet in input_lines]
        with torch.no_grad():
            answers = None
            for input_batch in self.__get_batch__(input_lines):
                tweet_emb = self.__bert_encoding__(self.bert_model, input_batch)  # [BATCH, 768]
                predictions = self.sigmoid(self.fc1(tweet_emb).float())  # [BATCH, NUM_CLASSES]
                if answers is None:
                    answers = predictions
                else:
                    answers = torch.cat([answers, predictions.detach()], dim=0)
            return np.array(answers.cpu())

    def __get_batch__(self, full_input):
        for i in range(0, len(full_input), BATCH_SIZE):
            yield full_input[i:min(i + BATCH_SIZE, len(full_input))]
