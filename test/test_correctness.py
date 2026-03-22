from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.models import GeminiModel

def test_correctness():
    # Initialize the judge using the model you actually have quota for
    gemini_judge = GeminiModel(
        model="gemini-2.5-flash",
        api_key="API-KEY"
    )

    correctness_metric = GEval(
        name='Correctness',
        criteria='check if the actual output is exactly the same as the expected output. If not return 0 else 1.',
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.1,
        model=gemini_judge
    )

    test_case = LLMTestCase(
        input="1+2",
        expected_output="3",
        actual_output="3"
    )

    assert_test(test_case, [correctness_metric])