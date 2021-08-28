from os import path
import random
import numpy as np
import torch
import argparse
from DataPrep import MNLIPrep
from transformers import BertForSequenceClassification, BertConfig

DATASET = 'mnli'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example_index",
                        default=16,
                        type=int,
                        help="Get attr output of the target example.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Prepare the data
    data_prep = MNLIPrep(f'{DATASET}_data')
    bert_input = data_prep.create_bert_input_dict('dev_matched',
                                                  slice(args.example_index, args.example_index+1))

    # Load a fine-tuned model
    state_dict = torch.load(path.join('models', f'model.{DATASET}.bin'), map_location=device)
    config = BertConfig(vocab_size=28996, num_labels=len(data_prep.labels()))
    model = BertForSequenceClassification\
        .from_pretrained('bert-base-cased',
                         state_dict=state_dict,
                         config=config).to(device)
    model.eval()

    res_attr = []
    att_all = []
    num_head, num_layer = 12, 12

    # bert input: dict keys: input_ids, attention_mask, token_type_ids, gold_labels
    # only one instance


if __name__ == '__main__':
    main()
