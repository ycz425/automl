import asyncio
import subprocess
import sys
import os


def _env_python(env_dir: str) -> str:
    if os.name == "nt":
        return os.path.join(env_dir, "Scripts", "python.exe")
    return os.path.join(env_dir, "bin", "python")


async def create_venv(env_dir: str, requirements_path: str) -> str:
    await asyncio.to_thread(
        subprocess.run,
        [sys.executable, '-m', 'venv', '--clear', env_dir],
        check=True,
        capture_output=True,
        text=True
    )

    env_python = _env_python(env_dir)

    await asyncio.to_thread(
        subprocess.run,
        [env_python, '-m', 'pip', 'install', '-r', requirements_path],
        check=True,
        capture_output=True,
        text=True
    )

    return env_python


async def run_train_script(env_python: str, script_path: str, input_path: str, output_path: str) -> None:
    await asyncio.to_thread(
        subprocess.run,
        [env_python, script_path, '--input', input_path, '--output', output_path],
        check=True,
        capture_output=True,
        text=True
    )


async def run_predict_script(env_python: str, script_path: str, model_path: str, input_path: str, output_path: str, threshold_path: str | None = None) -> None:
    args = [env_python, script_path, '--model', model_path, '--input', input_path, '--output', output_path]
    if threshold_path is not None:
        args += ['--threshold', threshold_path]

    await asyncio.to_thread(
        subprocess.run,
        args,
        check=True,
        capture_output=True,
        text=True
    )
