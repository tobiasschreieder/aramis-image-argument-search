import json
import logging
from pathlib import Path

log = logging.getLogger('Config')


class Config:

    data_dir: Path = Path('data/')
    output_dir: Path = Path('out/')
    working_dir: Path = Path('working/')
    data_image_format: bool = False
    on_win: bool = False

    _save_path = Path('config.json')

    _cfg = None

    @classmethod
    def get(cls) -> 'Config':
        cfg = cls()
        if Config._cfg is not None:
            return Config._cfg
        if Config._save_path.exists():
            try:
                with open(Config._save_path, 'r') as f:
                    cfg_json = json.load(f)
                cfg.data_dir = Path(cfg_json.get('data_dir', cfg.data_dir))
                cfg.output_dir = Path(cfg_json.get('output_dir', cfg.output_dir))
                cfg.working_dir = Path(cfg_json.get('working_dir', cfg.working_dir))
                cfg.data_image_format = bool(cfg_json.get('image_format', cfg.data_image_format))
                cfg.on_win = bool(cfg_json.get('on_win', cfg.on_win))
            except json.JSONDecodeError:
                pass
        log.debug('Config loaded')

        cfg.save()
        Config._cfg = cfg
        return cfg

    def save(self) -> None:
        log.debug('Config saved.')
        with open(Config._save_path, 'w+') as f:
            json.dump(self.to_dict(), f)

    def to_dict(self) -> dict:
        return {
            'data_dir': str(self.data_dir),
            'output_dir': str(self.output_dir),
            'working_dir': str(self.working_dir),
            'image_format': self.data_image_format,
            'on_win': self.on_win
        }
