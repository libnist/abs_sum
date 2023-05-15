import torch
from torch import nn


class GreedySummarizer(nn.Module):

    def __init__(self,
                 model,
                 src_tokenizer,
                 tgt_tokenizer,
                 src_max_token,
                 tgt_max_token,
                 device):

        super().__init__()

        self.model = model
        self.model.eval()

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.src_max_token = src_max_token
        self.tgt_max_token = tgt_max_token

        self.device = device

        self.end_token = tgt_tokenizer.token_to_id("<end>")

    def forward(self, input):
        assert isinstance(input, str)

        input_tokens = self.src_tokenizer.encode(input)[0]
        if len(input_tokens) < self.src_max_token:
            input_tokens += ([self.src_tokenizer.token_to_id("<pad>")] *
                             (self.src_max_token - len(input_tokens)))
        else:
            input_tokens = input_tokens[:self.src_max_token]
        input_tokens = torch.LongTensor([input_tokens]).to(self.device)

        output = [self.tgt_tokenizer.token_to_id("<start>")]

        for _ in range(self.tgt_max_token):
            dec_input = torch.LongTensor([output]).to(self.device)
            logits = self.model(input_tokens,
                                dec_input)
            pred = logits[:, -1:, :].softmax(dim=-1).argmax(dim=-1).item()
            output.append(pred)
            if pred == self.end_token:
                break
        return self.tgt_tokenizer.decode(output)
