from app.services.status_store import StatusStore, automl_status, automl_node
from app.services.file_storage import FileStorage
from dataclasses import dataclass
from langgraph.runtime import Runtime


@dataclass
class AutoMLContext:
    status_store: StatusStore
    file_storage: FileStorage


async def report_status(
    runtime: "Runtime[AutoMLContext]",
    status: automl_status | None = None,
    node: automl_node | None = None,
    message: str | None = None,
) -> None:
    await runtime.context.status_store.update(
        runtime.execution_info.thread_id, status=status, node=node, message=message
    )