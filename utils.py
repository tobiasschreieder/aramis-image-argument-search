import os
import shutil
import tempfile

from joblib import dump, load


class MemmappStore:

    def __init__(self):
        self.temp_folder = tempfile.mkdtemp()

    def store_in_memmap(self, value_to_store, name: str):
        filename = os.path.join(self.temp_folder, 'joblib_{}.mmap'.format(name))
        if os.path.exists(filename):
            os.unlink(filename)
        _ = dump(value_to_store, filename)
        return load(filename, mmap_mode='r+')

    def cleanup(self):
        try:
            shutil.rmtree(self.temp_folder)
        except OSError:
            pass  # this can sometimes fail under Windows
