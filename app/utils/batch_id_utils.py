import json
import os
from threading import Lock

class BatchIDTracker:
    def __init__(self, filepath="batch_ids.json"):
        self.filepath = filepath
        self.lock = Lock()
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w") as f:
                json.dump({}, f)

    def _load_data(self):
        with open(self.filepath, "r") as f:
            return json.load(f)

    def _save_data(self, data):
        with open(self.filepath, "w") as f:
            json.dump(data, f)

    def get_next_batch_id(self, username, camera_number):
        key = f"{username}:{camera_number}"
        with self.lock:
            data = self._load_data()
            current_id = data.get(key, -1) + 1
            data[key] = current_id
            self._save_data(data)
        return current_id
