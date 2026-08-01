from dataclasses import dataclass
from typing import Literal
import asyncio

type automl_status = Literal[
    'running',
    'need_clarification',
    'completed',
    'failed'
]


type automl_node = Literal[
    'prompt_agent',
    'data_agent',
    'plan_agent',
    'experiment_agent',
    'output_agent'
]

@dataclass
class Status:
    status: Literal[automl_status]
    node: Literal[automl_node]
    message: str | None = None


class StatusStore:
    def __init__(self):
        self._statuses: dict[str, Status] = {}
        self._events: dict[str, asyncio.Event] = {}

    async def update(self, thread_id: str, status: automl_status | None = None, node: automl_node | None = None, message: str | None = None):
        if thread_id not in self._statuses:
            if status is None or node is None:
                raise ValueError
            self._statuses[thread_id] = Status(status=status, node=node, message=message)
        else:
            if status is not None:
                self._statuses[thread_id].status = status
            if node is not None:
                self._statuses[thread_id].node = node
            if message is not None:
                self._statuses[thread_id].message = message

        self.get_event(thread_id).set()

    async def get_status(self, thread_id: str):
        status = self._statuses.get(thread_id)
        return status.status if status else None

    async def get_node(self, thread_id: str):
        status = self._statuses.get(thread_id)
        return status.node if status else None

    async def get_message(self, thread_id: str):
        status = self._statuses.get(thread_id)
        return status.message if status else None

    def get_event(self, thread_id: str):
        if thread_id not in self._events:
            self._events[thread_id] = asyncio.Event()
        return self._events[thread_id]