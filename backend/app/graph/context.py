from app.services.status_store import StatusStore
from app.services.file_storage import FileStorage
from dataclasses import dataclass


@dataclass
class AutoMLContext:
    status_store: StatusStore
    file_storage: FileStorage