from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from tema_3_evaluation.groq_llm import GroqDeepEval
from tema_3_evaluation.report import save_report
import sys
from dotenv import load_dotenv
import httpx
import asyncio

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

BASE_URL = "http://127.0.0.1:8000"
THRESHOLD = 0.8

test_cases = [
    # ToDo: Adăugați un scenariu care să fie evaluat de LLM as a Judge
    LLMTestCase(
        input=""
    ),
    # ToDo: Adăugați un scenariu care să fie evaluat de LLM as a Judge
    LLMTestCase(
        input=""
    ),
    # ToDo: Adăugați un scenariu care să fie evaluat de LLM as a Judge
    LLMTestCase(
        input=""
    ),
]

groq_model = GroqDeepEval()

evaluator1 = GEval(
    # ToDo: Adăugați numele metricii și criteriul de evaluare.
    name="",
    criteria="""    
    """,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=groq_model,
)

evaluator2 = GEval(
    # ToDo: Adăugați numele metricii și criteriul de evaluare.
    name="",
    criteria="""    
    """,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=groq_model,
)


async def _fetch_response(client: httpx.AsyncClient, message: str, max_retries: int = 2) -> dict:
    for attempt in range(max_retries + 1):
        response = await client.post(f"{BASE_URL}/chat/", json={"message": message})
        data = response.json()
        if data.get("detail") != "Raspunsul de chat a expirat":
            return data
        if attempt < max_retries:
            await asyncio.sleep(2)
    return data


async def _run_evaluation() -> tuple[list[dict], list[float], list[float]]:
    results: list[dict] = []
    scores1: list[float] = []
    scores2: list[float] = []

    async with httpx.AsyncClient(timeout=90.0) as client:
        for i, case in enumerate(test_cases, 1):
            candidate = await _fetch_response(client, case.input)
            case.actual_output = candidate

            evaluator1.measure(case)
            evaluator2.measure(case)

            print(f"[{i}/{len(test_cases)}] {case.input[:60]}...")
            # ToDo: Personalizați afișarea scorurilor pentru fiecare metrică.
            print(f"  #ToDo: {evaluator1.score:.2f} | #ToDo: {evaluator2.score:.2f}")

            results.append({
                "input": case.input,
                "response": candidate.get("response", str(candidate)) if isinstance(candidate, dict) else str(candidate),
                # ToDo: Adăugați în dicționar scorurile și motivele pentru fiecare metrică.
                "#ToDo_score": evaluator1.score,
                "#ToDo_reason": evaluator1.reason,
                "#ToDo_score": evaluator2.score,
                "#ToDo_reason": evaluator2.reason,
            })
            scores1.append(evaluator1.score)
            scores2.append(evaluator2.score)

    return results, scores1, scores2


def run_evaluation() -> None:
    results, scores1, scores2 = asyncio.run(_run_evaluation())
    output_file = save_report(results, scores1, scores2, THRESHOLD)
    print(f"\nRaport salvat in: {output_file}")


if __name__ == "__main__":
    run_evaluation()
