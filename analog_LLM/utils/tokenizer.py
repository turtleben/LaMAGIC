from transformers import PreTrainedTokenizer
import json
import torch
import os

class CustomTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file, eos_token="</s>", unk_token="<unk>", pad_token="<pad>", extra_ids=10, **kwargs):
        super().__init__(**kwargs)
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
        self.ids_to_tokens = {id_: token for token, id_ in self.vocab.items()}
        
        if extra_ids > 0:
            extra_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]
            self._extra_ids = extra_ids
            additional_special_tokens = extra_tokens
            # add additional_special_tokens to vocab
            for i, token in enumerate(additional_special_tokens):
                self.vocab[token] = len(self.vocab)
                self.ids_to_tokens[len(self.vocab)] = token

        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            # extra_ids=extra_ids,
            # additional_special_tokens=additional_special_tokens,
        )

        print('eos_token_id', self.eos_token_id)
        print('unk_token_id', self.unk_token_id)
        print('pad_token_id', self.pad_token_id)
        print('eos_token', self.eos_token)
        print(self.ids_to_tokens.values())
        self.sep_token_id = None
    
    def _tokenize(self, text):
        # Implement your tokenization logic here
        # This is a simple example that splits text by spaces
        tokens = text.split()
        return tokens

    def _convert_token_to_id(self, token):
        return self.vocab.get(token)

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.eos_token)

    def encode(self, text, add_special_tokens=True, **kwargs):
        tokens = self._tokenize(text)
        token_ids = [self._convert_token_to_id(token) for token in tokens]
        # print(token_ids)
        
        if add_special_tokens:
            # Example: Add a [CLS] token at the beginning and an [SEP] token at the end
            token_ids = token_ids + [self.eos_token_id]
        return  torch.tensor(token_ids)

    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        # print(token_ids)
        tokens = [self._convert_id_to_token(int(id_)) for id_ in token_ids]
        return " ".join(tokens)

    @property
    def vocab_size(self):
        return len(set(self.vocab.keys()))

    def save_vocabulary(self, save_directory: str, filename_prefix = None):
        out_vocab_file = os.path.join(
            save_directory, 'vocab.json'
        )

        return (out_vocab_file,)
