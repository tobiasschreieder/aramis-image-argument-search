import json
import logging
from pathlib import Path

log = logging.getLogger('Config')


class Config:

    data_location: Path = Path('data/')

    _save_path = Path('config.json')

    @classmethod
    def get(cls) -> 'Config':
        cfg = cls()
        if Config._save_path.exists():
            with open(Config._save_path, 'r') as f:
                cfg_json = json.load(f)
            cfg.data_location = Path(cfg_json.get('data_location', cfg.data_location))
        log.debug('Config loaded')

        cfg.save()
        return cfg

    def save(self) -> None:
        log.debug('Config saved.')
        with open(Config._save_path, 'w+') as f:
            json.dump(self.to_dict(), f)

    def to_dict(self) -> dict:
        return {
            'data_location': str(self.data_location),
        }
