from glob import glob 
import json
import itertools

from pyterrier import Transformer
import pandas as pd

def dict_to_text(source_col = "vector", target_col = "text") -> Transformer:

    import pyterrier as pt
    def _make_text_row(row):
        term_weights = row[source_col]
        repeated_terms = [ f"{term} "*tf for term, tf in term_weights.items()]
        return "".join(repeated_terms).strip()

    def _make_text_df(df : pd.DataFrame):
        df[target_col] = df.apply(_make_text_row, axis=1)
        return df

    return pt.apply.generic(_make_text_df)

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
                if len(batch_texts) == batch_size:
                    batch_term_weights = encoder.encode(batch_texts)
                    for tw in batch_term_weights:
                        yield tw
                    batch_texts = []
            if len(batch_texts) > 0:
                batch_term_weights = encoder.encode(batch_texts)
                for tw in batch_term_weights:
                    yield tw
        generator = cls(live_gen())
        return generator

    def texts_iter(self):
        for text in self.sparse_texts_iter:
            did = text["id"]
            term_weights = text["vector"]
            repeated_terms = [ f"{term} "*tf for term, tf in term_weights.items()]
            text = "".join(repeated_terms).strip()
            yield {"docno": did, "text": text}
