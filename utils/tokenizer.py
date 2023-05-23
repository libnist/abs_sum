from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer
)

from transformers import AutoTokenizer

from tqdm.auto import tqdm
import re
import numpy as np

from typing import List, Tuple


class CustomTokenizer:
    def __init__(self, 
                 vocab_size: int = 32000) -> Tokenizer:
        """Creates a WordPiece tokenizer and trains it based on given corpus
        and vocab_size.

        Args:
            vocab_size (int, optional): maximum vocab size. Defaults to 32_000.

        Returns:
            Tokenizer: Tokenizer
        """
        # Creating the tokenizer object
        self._tokenizer = Tokenizer(models.WordPiece(unk_token="<unk>"))

        # Adding a normalizer to the tokenizer
        self._tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)

        # Adding a pre_tokenizer to the tokenzier
        self._tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        # Create a trainer
        special_tokens = ["<pad>", "<start>", "<end>", "<sep>", "<unk>"]
        self._trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size, 
            special_tokens=special_tokens)
        
        # Add a decoder to the tokenizer
        self._tokenizer.decoder = decoders.WordPiece(prefix="##")
        
    def _set_post_processor(self):
        # Adding the post_processor to the tokenizer
        self._start_token_id = self._tokenizer.token_to_id("<start>")
        self._end_token_id = self._tokenizer.token_to_id("<end>")
        self._sep_token_id = self._tokenizer.token_to_id("<sep>")
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"<start>:0 $A:0 <end>:0",
            pair=f"<start>:0 $A:0 <sep>:0 $B:1 <end>:1",
            special_tokens=[("<start>", self._start_token_id),
                            ("<end>", self._end_token_id),
                            ("<sep>", self._sep_token_id)]
        )
    def get_training_corpus(self, 
                            training_dataset:list,
                            batch_size:int):
        for i in tqdm(range(0, len(training_dataset), batch_size)):
            yield training_dataset[i:i+batch_size]
            
    def fit(self, 
            training_dataset:List[str], 
            batch_size:int=10000):
        """Train the tokenizer given the training corpus.

        Args:
            training_dataset (List[str]): A list of strings containing the
            training corpus.
            batch_size (int, optional): Training batch size. Defaults to 10000.
        """
        self._tokenizer.train_from_iterator(
            self.get_training_corpus(training_dataset=training_dataset,
                                     batch_size=batch_size),
            trainer=self._trainer
        )
        self._set_post_processor()
        

    def save(self,
             path:str):
        """Save the tokenzier.

        Args:
            path (str): Path to save the tokenzier into a .json file.
        """
        self._tokenizer.save(path)
    
    @classmethod
    def load(cls,
             path:str):
        """Load a trained tokenizer.

        Args:
            path (str): Path to a saved tokenizer .json file.
        """
        tokenizer = cls()
        tokenizer._tokenizer = Tokenizer.from_file(path)
        tokenizer._set_post_processor()
        return tokenizer
        
    def get_type_ids(self,
                     ids:List[int]) -> list:
        """Return type_ids from list of ids.

        Args:
            ids (List[int]): A list of ids.

        Returns:
            _type_: list
        """
        ids = np.array(ids)
        sep_indices = np.where(ids==self._sep_token_id)[0]
        if len(sep_indices) == 0:
            type_ids = [0] * len(ids)
        else:
            last_index = sep_indices[-1]
            sep_indices[1:] -= sep_indices[:-1]
            type_ids = []
            for i, index in enumerate(sep_indices):
                type_ids += [i] * index
            type_ids += [i+1] * (len(ids) - last_index)
        return type_ids
    
    def encode(self,
               input:str) -> Tuple[List[int], List[int]]:
        """Given an string input, returns a tuple in form (ids, type_ids).

        Args:
            input (str): String input.

        Returns:
            Tuple: Tuple[List[int], List[int]]
        """
        input = re.sub(r"\\n", r".", input)
        input = re.sub(r"(\!|\?|\.) +([A-Z])", r"\1 <sep> \2", input)
        output = self._tokenizer.encode(input)
        return output.ids, self.get_type_ids(output.ids)
    
    def decode(self,
               input:List[int]) -> str:
        """Given ids, returns string corresponding to the ids.

        Args:
            input (List[int]): Input ids.

        Returns:
            str: str
        """
        return self._tokenizer.decode(input)
    
    def token_to_id(self,
                    token:str) -> int:
        """Given a token, returns it's corresponding id.

        Args:
            token (str): String token.

        Returns:
            int: int
        """
        return self._tokenizer.token_to_id(token)
    
    @property
    def vocab_size(self):
        return self._tokenizer.get_vocab_size()

class PretrainedCustomTokenizer:
  def __init__(self, name):
    self._tokenizer = AutoTokenizer.from_pretrained(name)

  def encode(self, input):
    output = self._tokenizer(input)
    return output["input_ids"], output["attention_mask"]

  def decode(self, input):
    return self._tokenizer.decode(input, skip_special_tokens=True)
