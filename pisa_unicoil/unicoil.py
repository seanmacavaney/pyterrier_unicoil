import torch 
from torch import nn

from transformers import PreTrainedModel, TrainingArguments, BertConfig, BertModel, BertTokenizer
from transformers import AutoModel 

class UniCOILEncoder(PreTrainedModel):
    config_class = BertConfig
    base_model_preifx = 'coil_encoder'
    load_tf_weights = None 

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.config = config 
        self.bert = BertModel(config)
        self.tok_proj = torch.nn.Linear(config.hidden_size, 1)
        self.init_weights()
    
    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        self.bert.init_weights()
        self.tok_proj.apply(self.__init__weights)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        input_shape = input_ids.size()
        device = input_ids.device 
        outputs = self.bert(input_ids = input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state 
        tok_weights = self.tok_proj(sequence_output)
        tok_weights = torch.relu(tok_weights)
        return tok_weights

