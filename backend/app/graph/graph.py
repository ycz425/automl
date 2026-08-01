from app.graph.schemas.state import AutoMLState
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from app.graph.nodes.prompt_agent_node import prompt_agent_node, prompt_status_update_node
from app.graph.nodes.data_agent_node import data_agent_node, data_status_update_node
from app.graph.nodes.plan_agent_node import plan_agent_node, plan_status_update_node
from app.graph.nodes.experiment_agent_node import experiment_agent_node, experiment_status_update_node
from app.graph.nodes.output_agent_node import output_agent_node, output_status_update_node
from app.graph.nodes.clarification_agent_node import clarification_agent_node, clarification_status_update_node
from app.graph.nodes.success_node import success_node
from app.graph.nodes.failure_node import failure_node

from app.graph.routers.prompt_router import prompt_router
from app.graph.routers.data_router import data_router
from app.graph.routers.experiment_router import experiment_router
from app.graph.routers.clarification_router import clarification_router

from app.graph.context import AutoMLContext


def build_graph():
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("app.graph.schemas.user_request", "UserRequest"),
            ("app.graph.schemas.data_info", "DatasetProfile"),
            ("app.graph.schemas.data_info", "DatasetAnalysis"),
            ("app.graph.schemas.clarification_request", "ClarificationRequest"),
            ("app.graph.schemas.clarification_request", "GeneratedClarificationRequest"),
            ("app.graph.schemas.plan", "Plan"),
            ('app.graph.schemas.experiment', 'Experiment')
        ]
    )

    builder = StateGraph(AutoMLState, context_schema=AutoMLContext)

    
    builder.add_node('prompt_agent', prompt_agent_node)
    builder.add_node('prompt_status_update', prompt_status_update_node)
    builder.add_edge(START, 'prompt_status_update')
    builder.add_edge('prompt_status_update', 'prompt_agent')

    builder.add_node('data_agent', data_agent_node)
    builder.add_node('data_status_update', data_status_update_node)
    builder.add_edge('data_status_update', 'data_agent')

    builder.add_node('plan_agent', plan_agent_node)
    builder.add_node('plan_status_update', plan_status_update_node)
    builder.add_edge('plan_status_update', 'plan_agent')

    builder.add_node('experiment_agent', experiment_agent_node)
    builder.add_node('experiment_status_update', experiment_status_update_node)
    builder.add_edge('experiment_status_update', 'experiment_agent')

    builder.add_edge('plan_agent', 'experiment_status_update')

    builder.add_node('output_agent', output_agent_node)
    builder.add_node('output_status_update', output_status_update_node)
    builder.add_edge('output_status_update', 'output_agent')
    
    builder.add_node('clarification_agent', clarification_agent_node)
    builder.add_node('clarification_status_update', clarification_status_update_node)
    builder.add_edge('clarification_status_update', 'clarification_agent')

    builder.add_node('success', success_node)
    builder.add_edge('output_agent', 'success')
    builder.add_edge('success', END)

    builder.add_node('failure', failure_node)
    builder.add_edge('failure', END)


    builder.add_conditional_edges(
        'prompt_agent',
        prompt_router,
        {
            'continue': 'data_status_update',
            'request_clarification': 'clarification_status_update',
            'block': 'failure'
        }
    )


    builder.add_conditional_edges(
        'data_agent',
        data_router,
        {
            'continue': 'plan_status_update',
            'request_clarification': 'clarification_status_update',
            'block': 'failure'
        }
    )

    builder.add_conditional_edges(
        'clarification_agent',
        clarification_router,
        {
            'continue_plan': 'plan_status_update',
            'continue_data': 'data_status_update',
            'request_clarification': 'clarification_status_update',
            'block': 'failure'
        }
    )

    builder.add_conditional_edges(
        'experiment_agent',
        experiment_router,
        {
            'output': 'output_status_update',
            'replan': 'plan_status_update'
        }
    )

    checkpointer = InMemorySaver(serde=serde)
    graph = builder.compile(checkpointer=checkpointer)
    return graph

graph = build_graph()