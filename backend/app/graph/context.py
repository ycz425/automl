from app.services.status_store import LocalStatusStore
from app.services.local_file_storage import LocalFileStorage
from dataclasses import dataclass


@dataclass
class AutoMLContext:
    status_store: LocalStatusStore
    file_storage: LocalFileStorage