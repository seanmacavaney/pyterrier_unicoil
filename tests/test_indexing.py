import unittest 
class TestIndexing(unittest.TestCase):

    def test_indexing_from_weight_files(self):
        from pyterrier_pisa import PisaIndex 
        from pisa_sparse import SparseGenerator
        gen = SparseGenerator.from_weight_files("resources/unicoil_tf/*")
        indexer = PisaIndex("unicoil-index", stemmer=None, threads=10, overwrite=True)
        indexer.index(gen.texts_iter())
        self.assertTrue(indexer.built())
        self.assertEqual(indexer.num_docs(), 20)
        self.assertEqual(indexer.num_terms(), 523)

    def test_live_indexing(self):
        from pyterrier_pisa import PisaIndex 
        from pisa_sparse import SparseGenerator 
        from pisa_sparse.unicoil import UniCOILDocumentEncoder
        encoder = UniCOILDocumentEncoder("castorini/unicoil-msmarco-passage", device="cpu")
        docs = [{"docno": 1, "text": "hello world"}, {"docno": 2, "text": "how are you"}]
        gen = SparseGenerator.from_pretrained_encoder(encoder, docs)
        indexer = PisaIndex("unicoil-index", stemmer=None, threads=10, overwrite=True)
        indexer.index(gen.texts_iter())
        self.assertTrue(indexer.built())
        self.assertEqual(indexer.num_docs(), 2)
        self.assertEqual(indexer.num_terms(), 5)
if __name__ == '__main__':
    unittest.main()