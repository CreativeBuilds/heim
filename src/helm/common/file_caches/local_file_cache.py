import os
from typing import Callable

from helm.common.general import ensure_directory_exists, generate_unique_id
from .file_cache import FileCache


class LocalFileCache(FileCache):
    def __init__(self, base_path: str, file_extension: str):
        ensure_directory_exists(base_path)
        self._location: str = base_path
        self._file_extension: str = file_extension

    def store(self, compute: Callable[[], bytes]) -> str:
        """
        Stores the output of `compute` as a file at a unique path.
        Returns the file path.
        """
        file_path: str = self.get_unique_file_location()
        with open(file_path, "wb") as f:
            f.write(compute())

        return file_path

    def get_unique_file_location(self) -> str:
        """Generate an unique file name at `base_path`"""

        def generate_one() -> str:
            file_name: str = f"{generate_unique_id()}.{self._file_extension}"
            return os.path.join(self._location, file_name)

        file_path: str
        while True:
            file_path = generate_one()
            if not os.path.exists(file_path):
                break
        return file_path
