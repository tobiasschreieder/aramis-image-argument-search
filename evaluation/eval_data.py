import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config

cfg = Config.get()
log = logging.getLogger('Evaluation')


eval_file = cfg.working_dir.joinpath(Path('image_eval.txt'))
if eval_file.exists():
    df = pd.read_csv(eval_file, sep=' ')
else:
    df = pd.DataFrame(columns=['image_id', 'user', 'Topic', 'Topic_correct', 'Argumentative', 'Stance'])

df.astype(dtype={
    'image_id': pd.StringDtype(),
    'user': pd.StringDtype(),
    'Topic': np.int,
    'Topic_correct': np.bool,
    'Argumentative': pd.StringDtype(),
    'Stance': pd.StringDtype(),
})
df.set_index(['image_id', 'user', 'Topic'], inplace=True)


def get_df() -> pd.DataFrame:
    return df.copy()


def save_df():
    df.to_csv(eval_file, sep=' ')
