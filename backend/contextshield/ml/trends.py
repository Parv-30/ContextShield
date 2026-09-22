from collections import defaultdict

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


def bucket_time_series(messages: list[dict], bucket: str = "D") -> list[dict]:
    if not messages:
        return []

    counts = defaultdict(int)
    for msg in messages:
        period = pd.Timestamp(msg["timestamp"]).to_period(bucket)
        key = (str(period.start_time.date()), msg["category"], msg["language"])
        counts[key] += 1

    return [
        {"time_bucket": time_bucket, "category": category, "language": language, "message_count": count}
        for (time_bucket, category, language), count in counts.items()
    ]


class TopicAnalyzer:
    def __init__(self, embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.model = BERTopic(embedding_model=self.embedding_model, verbose=False)

    def fit_transform(self, texts: list[str]) -> list[dict]:
        topics, _ = self.model.fit_transform(texts)
        topic_info = self.model.get_topic_info()

        results = []
        for topic_id in topic_info["Topic"]:
            if topic_id == -1:
                continue
            terms = [term for term, _ in self.model.get_topic(topic_id)]
            examples = [
                text for text, t in zip(texts, topics) if t == topic_id
            ][:5]
            results.append(
                {"topic_id": int(topic_id), "terms": terms, "example_messages": examples}
            )
        return results
