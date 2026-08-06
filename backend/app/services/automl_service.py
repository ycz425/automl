from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from app.graph.schemas.state import AutoMLState
from app.graph.graph import graph
from app.graph.context import AutoMLContext
from app.services.status_store import StatusStore
from app.services.file_storage import FileStorage
from langsmith import traceable


class AutoMLService:
    def __init__(self, graph: CompiledStateGraph,):
        self.graph = graph
        self.status_store = StatusStore()
        self.file_storage = FileStorage()

    @traceable(name="automl_run", run_type="chain")
    async def start(self, user_input: str, dataset_id: str, thread_id: str):
        config = self._config(thread_id)
        initial_state = AutoMLState(
            user_input=user_input,
            dataset_id=dataset_id,
            verbose=True
        )
        await self.file_storage.make_run_directory(thread_id)
        try:
            await self.graph.ainvoke(initial_state, config=config, context=self._context())
        except:
            await self.status_store.update(
                thread_id,
                status='failed',
                message='Failed.'
            )
    
    @traceable(name="automl_resume", run_type="chain")
    async def resume(self, user_input: str, thread_id: str):
        config = self._config(thread_id)
        try:
            await self.graph.ainvoke(Command(resume=user_input), config=config, context=self._context())
        except:
            await self.status_store.update(
                thread_id,
                status='failed',
                message='Failed.'
            )


    async def get_status(self, thread_id: str):
        if not (await self.exists(thread_id)):
            return None
        status = await self.status_store.get_status(thread_id)
        node = await self.status_store.get_node(thread_id)
        message = await self.status_store.get_message(thread_id)

        return {
            'status': status,
            'node': node,
            'message': message
        }

    async def exists(self, thread_id: str):
        return (await self.graph.checkpointer.aget_tuple(self._config(thread_id))) is not None
        
    def _context(self):
        return AutoMLContext(status_store=self.status_store, file_storage=self.file_storage)
            
    @staticmethod
    def _config(thread_id: str):
        return {
            "configurable": {
                "thread_id": thread_id,
            }
        }

    

automl_service = AutoMLService(graph)