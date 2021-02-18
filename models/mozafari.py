import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from utils import preprocess_tweet


NUM_CLASSES = 2


class MozafariModel(nn.Module):
    def __init__(self, mozafari_model, batch_size, use_hashtags=True):
        super(MozafariModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.fc1 = weight_norm(nn.Linear(768, NUM_CLASSES, bias=True))
        self.dropout = nn.Dropout(p=0.1)
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.sigmoid = nn.Sigmoid()
        self.load_state_dict(torch.load(mozafari_model, map_location='cpu')['model_state_dict'])
        self.to(self.device)
        self.use_hashtags = use_hashtags

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

    def forward(self, input_args):
        input_lines = input_args[0]  # this model only takes the posts as input
        input_lines = [preprocess_tweet(tweet, use_hashtags=self.use_hashtags) for tweet in input_lines]
        with torch.no_grad():
            answers = None
            for input_batch in self.__get_batch__(input_lines):
                tweet_emb = self.__bert_encoding__(self.bert_model, input_batch)  # [BATCH, 768]
                predictions = self.sigmoid(self.fc1(tweet_emb).float())  # [BATCH, NUM_CLASSES]
                predictions = (predictions >= 0.5).long()
                if answers is None:
                    answers = predictions
                else:
                    answers = torch.cat([answers, predictions.detach()], dim=0)
            answers = torch.max(answers, axis=1)[0]
            return answers.cpu()

    def __get_batch__(self, full_input):
        for i in range(0, len(full_input), self.batch_size):
            yield full_input[i:min(i + self.batch_size, len(full_input))]
