from app.services.status_store import StatusStore
from app.services.local_file_storage import LocalFileStorage
from dataclasses import dataclass


@dataclass
class AutoMLContext:
    status_store: StatusStore
    file_storage: LocalFileStorage