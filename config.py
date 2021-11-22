from pathlib import Path


class Config:

    data_location: Path = Path('data/')

    @classmethod
    def get(cls) -> 'Config':
        # TODO
        return cls()

    def save(self) -> None:
        # TODO
        pass
