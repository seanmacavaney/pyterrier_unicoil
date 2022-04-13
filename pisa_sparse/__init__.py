from glob import glob 
import json
import itertools

class SparseGenerator: 

    def __init__(self, sparse_texts_iter):
        self.sparse_texts_iter = sparse_texts_iter

    @classmethod
    def from_weight_files(cls, fdir):
        def read_from_file():
            for fn in glob(fdir):
                with open(fn) as f:
                    for line in f:
                        yield json.loads(line)
        generator = cls(read_from_file())
        return generator
        
    @classmethod
    def from_pretrained_encoder(cls, encoder, texts_iter,  batch_size = 1):
        def live_gen():
            batch_texts = []
            for text_obj in texts_iter:
                batch_texts.append(text_obj)
                if len(batch_ids) == batch_size:
                    batch_term_weights = encoder.encode(batch_texts)
                    for tw in batch_term_weights:
                        yield tw
        generator = cls(live_gen())
        return generator

    def texts_iter(self):
        for text in self.sparse_texts_iter:
            print(text)
            did = text["id"]
            term_weights = text["vector"]
            repeated_terms = [ f"{term} "*tf for term, tf in term_weights.items()]
            text = "".join(repeated_terms).strip()
            yield {"docno": did, "text": text}
