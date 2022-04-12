from glob import glob 
import json
import itertools
from unicoil import UniCOILEncoder

class SparseGenerator: 
    def __init__(self, sparse_texts_iter):
        self.sparse_texts_iter = sparse_texts_iter

    @classmethod
    def from_weight_files(fdir: str):
        def read_from_file():
            for fn in glob(fdir):
                with open(fn) as f:
                    for line in f:
                        yield json.loads(line)
        generator = SparseGenerator(read_from_file())
        return generator
        
    @classmethod
    def from_pretrained_encoder(encoder, tokenizer, texts_iter,  batch_size = 1,  device = "cpu"):
        def live_gen():
            encoder = encoder.to(device)
            batch_texts = []
            for text_obj in texts_iter:
                batch_texts.append(text_obj)
                if len(batch_ids) == batch_size:
                    batch_term_weights = encoder.encode(batch_texts)
                    for tw in batch_term_weights
                        yield tw
        generator = SparseGenerator(live_gen())
        return generator

    def texts_iters(self)
        for text in self.sparse_texts_iter:
            did = text["id"]
            term_weights = text["vector"]
            repeated_terms = [ f"{term} "*tf for term, tf in term_weights.items()]
            text = "".join(repeated_terms).strip()
            yield {"docno": did, "text": text}
