from contextshield.db.models import Conversation, Message, MessageLabel, EscalationSnapshot
from contextshield.ml.escalation import compute_timeline


class ConversationService:
    def __init__(self, session, classifier, tau_low: float, tau_high: float, window: int = 3, alpha: float = 0.6):
        self.session = session
        self.classifier = classifier
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.window = window
        self.alpha = alpha

    def create_conversation(self, messages: list[dict], source: str, language: str) -> Conversation:
        texts = [m["text"] for m in messages]
        predictions = self.classifier.predict_batch(texts)

        convo = Conversation(source=source, language=language)
        self.session.add(convo)
        self.session.flush()

        toxicity_scores = []
        for idx, (msg, prediction) in enumerate(zip(messages, predictions)):
            db_message = Message(
                conversation_id=convo.id,
                seq_index=idx,
                text=msg["text"],
                language=language,
                author=msg.get("author"),
            )
            self.session.add(db_message)
            self.session.flush()

            for category, probability in prediction["probabilities"].items():
                self.session.add(
                    MessageLabel(
                        message_id=db_message.id,
                        category=category,
                        probability=probability,
                        toxicity_score=prediction["toxicity_score"],
                    )
                )
            toxicity_scores.append(prediction["toxicity_score"])

        timeline = compute_timeline(
            toxicity_scores, w=self.window, alpha=self.alpha, tau_low=self.tau_low, tau_high=self.tau_high,
        )
        for entry in timeline:
            self.session.add(
                EscalationSnapshot(
                    conversation_id=convo.id,
                    seq_index=entry["seq_index"],
                    windowed_mean=entry["windowed_mean"],
                    slope=entry["slope"],
                    escalation_score=entry["escalation_score"],
                    status=entry["status"],
                )
            )

        self.session.commit()
        self.session.refresh(convo)
        return convo

    def get_conversation(self, conversation_id: int) -> Conversation | None:
        return self.session.get(Conversation, conversation_id)

    def list_conversations(self, limit: int = 20, offset: int = 0) -> list[Conversation]:
        return (
            self.session.query(Conversation)
            .order_by(Conversation.created_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )
