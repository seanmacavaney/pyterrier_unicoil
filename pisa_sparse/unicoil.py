import torch 
from torch import nn

from transformers import PreTrainedModel, BertModel, BertTokenizer

class UniCOILEncoder(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config 
        self.bert = BertModel(config)
        self.tok_proj = torch.nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        input_shape = input_ids.size()
        device = input_ids.device 
        outputs = self.bert(input_ids = input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state 
        tok_weights = self.tok_proj(sequence_output)
        tok_weights = torch.relu(tok_weights)
        return tok_weights


class UniCOILDocumentEncoder:
    def __init__(self, model_name, tokenizer_name=None, device='cuda:0'):
        self.device = device
        self.model = UniCoilEncoder.from_pretrained(model_name)
        self.model.to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name or model_name)

    def encode(self, batch, fp16=False):
        texts = [b["text"] for b in batch]
        docnos = [b["docno"] for b in batch]
        max_length = 512  # hardcode for now
        inps = self.tokenizer(texts, max_length=max_length, padding='longest', truncation=True, add_special_tokens=True, return_tensors='pt').to(self.device)
        if fp16:
            with autocast():
                with torch.no_grad():
                    batch_weights = self.model(**inps).cpu().detach().numpy()
        else:
            batch_weights = self.model(**inps).cpu().detach().numpy()
        batch_token_ids = inps["input_ids"].cpu().detach().numpy()
        return self._output_to_weight_dicts(docnos, batch_token_ids, batch_weights)

    def _output_to_weight_dicts(self, docnodes, batch_token_ids, batch_weights):
        to_return = []
        for idx, token_ids in enumerate(batch_token_ids):
            weights = batch_weights[idx].flatten()
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids.flatten())
            tok_weights = {}
            for tok, weight in zip(tokens, weights):
                quantized_weight  = math.ceil(weight*100)
                if tok in ['[CLS]', 'PAD']:
                    continue
                if tok == '[PAD]':
                    break
                if tok not in tok_weights:
                    tok_weights[tok] = quantized_weight
                elif quantized_weight > tok_weights[tok]:
                    tok_weights[tok] = quantized_weight
            json_obj = {"id": docnos[idx], "vector": tok_weights}
            to_return.append(json_obj)
        return to_return

