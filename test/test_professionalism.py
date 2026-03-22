from deepeval import evaluate
from deepeval.test_case import ConversationalTestCase, Turn, TurnParams
from deepeval.metrics import ConversationalGEval
from deepeval.models import GeminiModel


def test_professionalism():
    gemini_judge = GeminiModel(
        model="gemini-2.5-flash",
        api_key="AIzaSyDTbHGh3NzSYLdmq7GqvRCiGu7MY-lqEo8"
    )

    professionalism_metric = ConversationalGEval(
        name="Professionalism",
        criteria="Determine whether the assistant answered the questions of the user in a professional and polite manner.",
        model=gemini_judge
    )

    conversation_example1 = ConversationalTestCase(
        turns=[
            Turn(role='user', content='What is the capital of India?'),
            Turn(role='assistant', content='New Delhi. You should know this basic fact.'),
            Turn(role='user', content='What is the currency used there?'),
            Turn(role='assistant', content='Rupees. Google it next time.')
        ]
    )

    conversation_example2 = ConversationalTestCase(
        turns=[
            Turn(role='user', content='What is the capital of India?'),
            Turn(role='assistant', content='The capital of India is New Delhi.'),
            Turn(role='user', content='What is the currency used there?'),
            Turn(role='assistant', content='The currency used in India is the Indian Rupee (INR).')
        ]
    )

    evaluate(
        test_cases=[conversation_example1, conversation_example2],
        metrics=[professionalism_metric]
    )