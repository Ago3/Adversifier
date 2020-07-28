# import fileinput
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

        self.racism_model = BertForSequenceClassification.from_pretrained('bert-base-uncased').eval()
        self.racism_model.load_state_dict(torch.load(racism_model, map_location='cpu'))
        self.racism_model.to(self.device)

        self.sexism_model = BertForSequenceClassification.from_pretrained('bert-base-uncased').eval()
        self.sexism_model.load_state_dict(torch.load(sexism_model, map_location='cpu'))
        self.sexism_model.to(self.device)

    def forward(self, input_lines):
        input_lines = [preprocess_tweet(tweet) for tweet in input_lines]
        predictions = None
        for batch_lines in tqdm(batcher(input_lines, batch_size=self.batch_size)):
            batch = self.tokenizer.batch_encode_plus(batch_lines, return_tensors='pt', padding='max_length')
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.no_grad():
                racism_predictions = torch.tensor(self.racism_model(**batch)[0].argmax(-1).tolist())
                sexism_predictions = torch.tensor(self.sexism_model(**batch)[0].argmax(-1).tolist())

            current_predictions = racism_predictions | sexism_predictions
            if predictions is None:
                predictions = current_predictions
            else:
                predictions = torch.cat([predictions, current_predictions], dim=0)
        print(predictions.shape)
        return predictions


# def kennedy_model():
#     parser = ArgumentParser()
#     parser.add_argument('racism_model')
#     parser.add_argument('sexism_model')
#     parser.add_argument('--input', default='-')
#     parser.add_argument('--labels_file', default='-')
#     parser.add_argument('--batch_size', default=100, type=int)
#     parser.add_argument('--device', default='cpu')
#     parser.add_argument('--model', default='kennedy')
#     args = parser.parse_args()

#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     racism_model = BertForSequenceClassification.from_pretrained('bert-base-uncased').eval()
#     racism_model.load_state_dict(torch.load(args.racism_model, map_location='cpu'))
#     racism_model.to(args.device)

#     sexism_model = BertForSequenceClassification.from_pretrained('bert-base-uncased').eval()
#     sexism_model.load_state_dict(torch.load(args.sexism_model, map_location='cpu'))
#     sexism_model.to(args.device)

#     predictions = None
#     labels = None

#     with open(args.labels_file, 'r') as f:
#         labels = torch.stack([_string_to_label_(line.split('\t')[2]) for line in f.readlines()], dim=0)

#     if args.model == 'kennedy':
#         for batch_lines in tqdm(batcher(fileinput.input([args.input]), batch_size=args.batch_size)):
#             batch = tokenizer.batch_encode_plus(batch_lines, return_tensors='pt', pad_to_max_length=True)
#             batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
#             with torch.no_grad():
#                 racism_predictions = torch.tensor(racism_model(**batch)[0].argmax(-1).tolist())
#                 sexism_predictions = torch.tensor(sexism_model(**batch)[0].argmax(-1).tolist())

#             current_predictions = torch.cat([racism_predictions.view([racism_predictions.shape[0], 1]), sexism_predictions.view([sexism_predictions.shape[0], 1])], dim=1)
#             if predictions is None:
#                 predictions = current_predictions
#             else:
#                 predictions = torch.cat([predictions, current_predictions], dim=0)
#     elif args.model == 'always_neither':
#         predictions = torch.zeros(labels.shape, device=args.device)
    
#     predictions_3c = __convert_to_3_classes__(predictions, args.device)
#     labels_3c = __convert_to_3_classes__(labels, args.device)

#     f1_3c = f1_score(labels_3c.cpu(), predictions_3c.cpu(), average='micro')
#     print("F1 score with 3 classes: {}".format(f1_3c))
#     micro_per_class_f1  =_compute_scores_per_classes_(predictions_3c.cpu(), labels_3c.cpu())
#     f1_3c_macro = f1_score(labels_3c.cpu(), predictions_3c.cpu(), average='macro')
#     print("MACRO F1 score with 3 classes: {}".format(f1_3c_macro))
#     _compute_scores_per_classes_(predictions_3c.cpu(), labels_3c.cpu(), average='macro')
#     f1_3c_w = f1_score(labels_3c.cpu(), predictions_3c.cpu(), average='weighted')
#     print("WEIGHTED F1 score with 3 classes: {}".format(f1_3c_w))
#     weighted_per_class_f1 = _compute_scores_per_classes_(predictions_3c.cpu(), labels_3c.cpu(), average='weighted')
#     micro_abusive_score = _compute_abusive_score_(predictions_3c.cpu(), labels_3c.cpu(), average='micro')
#     _compute_abusive_score_(predictions_3c.cpu(), labels_3c.cpu(), average='macro')
#     weighted_abusive_score = _compute_abusive_score_(predictions_3c.cpu(), labels_3c.cpu(), average='weighted')
#     print('SUMMARY: {}\t{}{}\t{}\t{}{}\n'.format(f1_3c, micro_per_class_f1, micro_abusive_score, f1_3c_w, weighted_per_class_f1, weighted_abusive_score))
