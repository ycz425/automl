import dotenv
from app.agents.research_agent import ResearchAgent
from app.graph.schemas.user_request import UserRequest
from app.graph.schemas.data_info import DatasetProfile, DatasetAnalysis
from langsmith import aevaluate

dotenv.load_dotenv()


async def run_agent(inputs: dict):
    research_agent = ResearchAgent()

    user_request = UserRequest.model_validate(inputs['user_request'])
    dataset_profile = DatasetProfile.model_validate(inputs['dataset_profile'])
    dataset_analysis = DatasetAnalysis.model_validate(inputs['dataset_analysis'])

    result = research_agent.search(user_request, dataset_profile, dataset_analysis)
    retrieved_sources = [point.source for point in result.points] if result else []

    return {'retrieved_sources': retrieved_sources}


def hit_rate(outputs: dict, reference_outputs: dict):
    expected = set(reference_outputs['expected_sources'])
    retrieved = outputs['retrieved_sources']

    hit = any(source in expected for source in retrieved)
    return {
        'key': 'hit_rate',
        'score': float(hit),
        'comment': 'No expected source retrieved.' if not hit else None
    }


def recall_at_k(outputs: dict, reference_outputs: dict):
    expected = set(reference_outputs['expected_sources'])
    if not expected:
        return

    retrieved = set(outputs['retrieved_sources'])
    found = expected & retrieved
    return {
        'key': 'recall_at_k',
        'score': len(found) / len(expected),
        'comment': f'Missed: {sorted(expected - found)}' if found != expected else None
    }


def mrr(outputs: dict, reference_outputs: dict):
    expected = set(reference_outputs['expected_sources'])
    retrieved = outputs['retrieved_sources']

    for rank, source in enumerate(retrieved, start=1):
        if source in expected:
            return {'key': 'mrr', 'score': 1 / rank}
    return {'key': 'mrr', 'score': 0.0}


if __name__ == '__main__':
    import asyncio

    asyncio.run(aevaluate(
        run_agent,
        data='automl_research',
        evaluators=[
            hit_rate,
            recall_at_k,
            mrr
        ],
        experiment_prefix='automl_research_eval'
    ))
