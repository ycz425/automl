from fastapi import UploadFile
import pandas as pd
import io

CHUNK_SIZE = 1024 * 1024


async def upload_file_to_dataframe(file: UploadFile) -> pd.DataFrame:
    await file.seek(0)
    chunks: list[bytes] = []
    while True:
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            break
        chunks.append(chunk)
    return pd.read_csv(io.BytesIO(b"".join(chunks)))
