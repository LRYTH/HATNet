from .ciderD_scorer import CiderScorer   # 原来是相对路径，Python 3 不变

class CiderD:
    def __init__(self, n=4, sigma=6.0, df="corpus"):
        self._n = n
        self._sigma = sigma
        self._df = df

    def compute_score(self, gts, res):
        cider_scorer = CiderScorer(n=self._n)
        for res_id in res:
            hypo = res_id['caption']
            ref = gts[res_id['image_id']]
            assert type(hypo) is list
            assert len(hypo) == 1
            assert type(ref) is list
            assert len(ref) > 0
            cider_scorer += (hypo[0], ref)
        (score, scores) = cider_scorer.compute_score(self._df)
        return score, scores

    def method(self):
        return "CIDEr-D"