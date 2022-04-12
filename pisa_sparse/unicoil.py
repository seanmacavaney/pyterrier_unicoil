import torch 
from torch import nn

from transformers import PreTrainedModel, TrainingArguments, BertConfig, BertModel, BertTokenizer
from transformers import AutoModel 

class UniCOILEncoder(PreTrainedModel):
    def __init__(self, config: BertConfig):
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

    def encode(self, texts, fp16=False):
        max_length = 512  # hardcode for now
        input_ids = self.tokenizer(texts, max_length=max_length, padding='longest', truncation=True, add_special_tokens=True, return_tensors='pt').to(self.device)["input_ids"]
        if fp16:
            with autocast():
                with torch.no_grad():
                    batch_weights = self.model(input_ids).cpu().detach().numpy()
        else:
            batch_weights = self.model(input_ids).cpu().detach().numpy()
        batch_token_ids = input_ids.cpu().detach().numpy()
        return self._output_to_weight_dicts(batch_token_ids, batch_weights)

    def _output_to_weight_dicts(self, batch_token_ids, batch_weights):
        to_return = []
        for i in range(len(batch_token_ids)):
            weights = batch_weights[i].flatten()
            tokens = self.tokenizer.convert_ids_to_tokens(batch_token_ids[i])
            tok_weights = {}
            for j in range(len(tokens)):
                tok = str(tokens[j])
                weight = float(weights[j])
                if tok in ['[CLS]', 'PAD']:
                    continue
                if tok == '[PAD]':
                    break
                if tok not in tok_weights:
                    tok_weights[tok] = weight
                elif weight > tok_weights[tok]:
                    tok_weights[tok] = weight
            to_return.append(tok_weights)
        return to_return

