import unittest 
import tempfile

class TestIndexing(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        #super(TestIndexing, self).__init__(*args, **kwargs)
        import pyterrier as pt
        if not pt.started():
            pt.init()

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        try:
            shutil.rmtree(self.test_dir)
        except:
            pass

    def test_vaswani(self):
        import pyterrier as pt
        from pyterrier_sparse import dict_to_text
        from pyterrier_sparse.unicoil import UniCOILDocumentEncoder
        encoder = UniCOILDocumentEncoder("castorini/unicoil-msmarco-passage", device="cpu")

        dataset = pt.get_dataset("vaswani")
        docs = dataset.get_corpus_iter()
        target_indexer = pt.IterDictIndexer(self.test_dir, overwrite=True)
        target_indexer.setProperty("termpipelines", "")

        indexer = encoder >> dict_to_text() >> target_indexer
        ref = indexer.index(docs)
        df = pt.Experiment([
                pt.BatchRetrieve(ref, wmodel='BM25'),
                pt.BatchRetrieve.from_dataset(dataset, wmodel='BM25')
            ],
            dataset.get_topics(),
            dataset.get_qrels(),
            ["map", "mrr"],
            names=['unicoil bm25', 'bm25']
            )
        print(df)