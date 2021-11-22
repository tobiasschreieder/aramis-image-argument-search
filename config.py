from pathlib import Path


class Config:

    data_location: Path = Path('data/')
    # data_location: Path = Path('G:/IR Datensatz/')

    @classmethod
    def get(cls) -> 'Config':
        # TODO
        return cls()

    def save(self) -> None:
        # TODO
        pass
