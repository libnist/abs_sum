from transformers import AutoModelForSeq2SeqLM

from ..models import get_attn_mask

class PretrainedModel:
  def __init__(self, name, pad_id, requires_grad=False):
    self.pad_id = pad_id
    self.model = AutoModelForSeq2SeqLM.from_pretrained(name)
    for param in self.model.parameters():
      param.requires_grad = requires_grad

  def __call__(self,
               src,
               tgt):
    src_mask = (src == self.pad_id)
    tgt_mask = (tgt == self.pad_id))
    output = self.model(input_ids=src,
                        decoder_input_ids=tgt,
                        attention_mask=src_mask,
                        decoder_attention_mask=tgt_mask
                        )
    return output.encoder_last_hidden_state, output.logits
  
  def eval(self):
    self.model.eval()