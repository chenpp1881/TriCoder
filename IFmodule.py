import pdb
import torch
import random
import torch.nn as nn
from transformers import T5EncoderModel, RobertaModel
from modules.Multihead_Attention import DecoderAttention
from modules.SubLayerConnection import SublayerConnection


class InformationFusionBlock(nn.Module):

    def __init__(self, hidden, dropout, args, num_layers):
        super().__init__()
        self.args = args

        if args.model_path == 'graphcodebert-base':
            print(f"==== model_path = {args.model_path} ====")
            self.codeT5 = RobertaModel.from_pretrained(args.model_path)
            self.model_type = 'roberta'
        elif args.model_path == 'codebert-base':
            print(f"==== model_path = {args.model_path} ====")
            self.codeT5 = RobertaModel.from_pretrained(args.model_path)
            self.model_type = 'roberta'
        elif args.model_path == 'unixcoder-base':
            print(f"==== model_path = {args.model_path} ====")
            self.codeT5 = RobertaModel.from_pretrained(args.model_path)
            self.model_type = 'roberta'
        elif args.model_path == 'codet5-base':
            print(f"==== model_path = {args.model_path} ====")
            self.codeT5 = T5EncoderModel.from_pretrained(args.model_path)
            self.model_type = 't5'
        else:
            self.codeT5 = T5EncoderModel.from_pretrained(args.model_path)
            self.model_type = 't5'

        if args.task_id in ["SCVD", "Devign", "Reveal"]:
            num_classes = 2
        elif args.task_id == "DefectPre":
            num_classes = 4
        elif args.task_id == "POJ-104":
            num_classes = 104
        elif args.task_id == "Authorship":
            num_classes = 66

        self.fc = nn.Linear(hidden, num_classes)
        self.self_attention = nn.ModuleList()
        self.sublayer_connection1 = nn.ModuleList()
        self.linear_layers = nn.ModuleList()

        self.num_layers = num_layers

        for _ in range(self.num_layers):
            self.self_attention.append(DecoderAttention(d_model=hidden))
            self.sublayer_connection1.append(SublayerConnection(size=hidden, dropout=dropout))
            self.linear_layers.append(nn.Linear(in_features=hidden, out_features=hidden))

    def _encode_tokens(self, tokens):
        if self.model_type == 'roberta':
            return self.codeT5(
                input_ids=tokens.get('input_ids'),
                attention_mask=tokens.get('attention_mask')
            ).last_hidden_state
        else:
            return self.codeT5(**tokens).last_hidden_state


    def forward(self, code_tokens, desc_tokens, woc=False):
        code_attention_mask = code_tokens['attention_mask']
        code_embeddings = self._encode_tokens(code_tokens)
        ini_emb = code_embeddings

        desc_embeddings = []
        desc_attention_masks = []
        for des in desc_tokens:
            desc_embeddings.append(self._encode_tokens(des))
            desc_attention_masks.append(des['attention_mask'])

        if not desc_embeddings:
            mask = code_attention_mask.unsqueeze(-1).expand(code_embeddings.size()).float()
            sum_embeddings = (code_embeddings * mask).sum(dim=1)
            valid_token_counts = mask.sum(dim=1).clamp(min=1)
            mean_embeddings = sum_embeddings / valid_token_counts
            return self.fc(mean_embeddings)

        combined = list(zip(desc_embeddings, desc_attention_masks))
        random.shuffle(combined)
        desc_embeddings, desc_attention_masks = zip(*combined)

        for layer_idx in range(len(desc_embeddings)):
            desc_attention_mask = desc_attention_masks[layer_idx]

            attention_mask = torch.bmm(
                code_attention_mask.unsqueeze(2).float(),
                desc_attention_mask.unsqueeze(1).float()
            )

            code_embeddings = self.sublayer_connection1[layer_idx](
                code_embeddings,
                lambda _code_embeddings: self.self_attention[layer_idx](
                    _code_embeddings, desc_embeddings[layer_idx], desc_embeddings[layer_idx], attention_mask
                )
            )
            code_embeddings = self.linear_layers[layer_idx](code_embeddings)

        if woc:
            mean_fused_embeddings = code_embeddings.mean(dim=1)
            return self.fc(mean_fused_embeddings)
        else:
            mask = code_attention_mask.unsqueeze(-1).expand(code_embeddings.size()).float()
            sum_embeddings = (code_embeddings * mask).sum(dim=1)
            valid_token_counts = mask.sum(dim=1).clamp(min=1)
            mean_embeddings = sum_embeddings / valid_token_counts

            sum_ini_emb = (ini_emb * mask).sum(dim=1)
            mean_ini_emb = sum_ini_emb / valid_token_counts

            return self.fc(mean_embeddings + mean_ini_emb)