from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models import GeminiModel

def test_rag():
    gemini_judge = GeminiModel(
        model="gemini-2.5-flash",
        api_key="AIzaSyDTbHGh3NzSYLdmq7GqvRCiGu7MY-lqEo8"
    )

    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7,
        model=gemini_judge,
        include_reason=True
    )
    
    answer_relevancy_metric = AnswerRelevancyMetric(
        threshold=0.7,
        model=gemini_judge,
        include_reason=True
    )

    test_case = LLMTestCase(
        input="What is the capital of India?",
        actual_output="New Delhi is the capital of India.",
        retrieval_context=["India is a country in South Asia. Its capital is New Delhi."]
    )

    assert_test(test_case, [faithfulness_metric, answer_relevancy_metric])