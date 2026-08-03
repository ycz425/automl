from dataclasses import dataclass
from typing import Literal
import asyncio
from google.cloud import firestore
import dotenv
import os

dotenv.load_dotenv()
GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

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


class LocalStatusStore:
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


class FirestoreStatusStore:
    def __init__(self):
        self.async_client = firestore.AsyncClient(project=GOOGLE_CLOUD_PROJECT)
        self.sync_client = firestore.Client(project=GOOGLE_CLOUD_PROJECT)
        self.async_collection = self.async_client.collection('runs')
        self.sync_collection = self.sync_client.collection('runs')

    async def update(self, thread_id: str, status: automl_status | None = None, node: automl_node | None = None, message: str | None = None):
        document = self.async_collection.document(thread_id)
        snapshot = await document.get()

        if not snapshot.exists:
            if status is None or node is None:
                raise ValueError
            await document.set({'status': status, 'node': node, 'message': message})
        else:
            status_update = {}
            if status is not None:
                status_update['status'] = status
            if node is not None:
                status_update['node'] = node
            if message is not None:
                status_update['message'] = message
            await document.update(status_update)

    async def get_status(self, thread_id: str):
        document = self.async_collection.document(thread_id)
        snapshot = await document.get()
        return snapshot.to_dict()['status'] if snapshot.exists else None

    async def get_node(self, thread_id: str):
        document = self.async_collection.document(thread_id)
        snapshot = await document.get()
        return snapshot.to_dict()['node'] if snapshot.exists else None

    async def get_message(self, thread_id: str):
        document = self.async_collection.document(thread_id)
        snapshot = await document.get()
        return snapshot.to_dict()['message'] if snapshot.exists else None

    async def delete(self, thread_id: str):
        document = self.async_collection.document(thread_id)
        await document.delete()

    async def subscribe(self, thread_id: str) -> tuple[asyncio.Event, firestore.Watch]:
        event = asyncio.Event()
        loop = asyncio.get_running_loop()
        def callback(snapshot, changes, read_time):
            if snapshot[0].exists:
                loop.call_soon_threadsafe(event.set)

        return event, self.sync_collection.document(thread_id).on_snapshot(callback)

    async def exists(self, thread_id: str):
        document = self.async_collection.document(thread_id)
        snapshot = await document.get()
        return snapshot.exists