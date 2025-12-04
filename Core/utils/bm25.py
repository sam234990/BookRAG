import math
import pickle
from collections import Counter
import tqdm
from typing import List


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        """
        BM25算法的构造器
        :param docs: 分词后的文档列表，每个文档是一个包含词汇的列表
        :param k1: BM25算法中的调节参数k1
        :param b: BM25算法中的调节参数b
        """
        self.original_docs = docs
        self.k1 = k1
        self.b = b
        # 内部进行分词处理
        self.tokenized_docs = [self._tokenize(doc) for doc in self.original_docs]

        self.doc_len = [len(doc) for doc in self.tokenized_docs]
        self.avgdl = (
            sum(self.doc_len) / len(self.tokenized_docs) if self.tokenized_docs else 0
        )
        self.doc_freqs = []
        self.idf = {}

    def _tokenize(self, text: str) -> List[str]:
        """
        一个简单的分词器。可以根据需要替换成更复杂的分词库。
        :param text: 输入的文本字符串。
        :return: 分词后的词汇列表。
        """
        # 简单的实现：转为小写并按空格分割。
        # 对于中文或更复杂的场景，可以替换为例如：jieba.lcut(text)
        return text.lower().split()

    def initialize(self):
        """
        初始化方法，计算所有词的逆文档频率。
        """
        if not self.tokenized_docs:
            return  # 如果没有文档，则不执行任何操作

        df = {}  # 存储每个词在多少不同文档中出现
        for doc in self.tokenized_docs:
            self.doc_freqs.append(Counter(doc))
            for word in set(doc):
                df[word] = df.get(word, 0) + 1

        # 计算每个词的IDF值
        num_docs = len(self.tokenized_docs)
        for word, freq in tqdm.tqdm(df.items(), total=len(df), desc="Calculating IDF"):
            self.idf[word] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1)

    def _get_score(self, doc_index: int, query_tokens: List[str]) -> float:
        """
        计算单个文档与查询的BM25得分。
        """
        score = 0.0
        doc_freq = self.doc_freqs[doc_index]
        for word in query_tokens:
            if word in doc_freq:
                freq = doc_freq[word]
                score += (self.idf.get(word, 0) * freq * (self.k1 + 1)) / (
                    freq
                    + self.k1
                    * (1 - self.b + self.b * self.doc_len[doc_index] / self.avgdl)
                )
        return score

    def search(self, query_text: str, top_k: int) -> List[dict]:
        """
        根据查询文本，返回最相关的 top_k 个原始文档。
        """
        # 使用与文档相同的分词方法处理查询
        query_tokens = self._tokenize(query_text)

        scores = [
            (self._get_score(i, query_tokens), i)
            for i in range(len(self.tokenized_docs))
        ]
        scores.sort(key=lambda x: x[0], reverse=True)

        top_docs = scores[:top_k]

        results = []
        for score, idx in top_docs:
            results.append(
                {
                    "id": idx,
                    "score": score,
                    # 直接从保存的原始文档列表中返回内容
                    "content": self.original_docs[idx],
                }
            )
        return results

    def save(self, path):
        """
        将BM25对象保存到本地文件
        :param path: 保存文件的路径
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """
        从本地文件加载BM25对象
        :param path: 保存文件的路径
        :return: 加载的BM25对象
        """
        with open(path, "rb") as f:
            bm25 = pickle.load(f)
        return bm25

    def close(self):
        """
        释放BM25对象的资源
        """
        del self.original_docs
