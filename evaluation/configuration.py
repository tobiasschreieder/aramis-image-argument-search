import dataclasses
import json


@dataclasses.dataclass
class Configuration:
    # RetrievalSystem:
    topic_weight: float = 0.3
    argument_weight: float = 0.4
    prefetch_top_k: float = 2

    # TopicRankingDirichlet
    tm_name = 'TopicRankingDirichlet'
    tq_alpha: int = 1000
    alpha: int = 1000
    #
    # StandardArgumentModel
    #   - usage of features
    am_name = 'StandardArgumentModel'
    arg_features = []
    #
    # StandardStanceModel
    #   - usage of features
    sm_name = 'StandardStanceModel'
    stance_features = []

    def to_json(self) -> str:
        d = {
            'retrieval_system': {
                'topic_weight': self.topic_weight,
                'argument_weight': self.argument_weight,
                'prefetch_top_k': self.prefetch_top_k,
            },
            'topic_model': {
                'name': self.tm_name,
                'tq_alpha': self.tq_alpha,
                'alpha': self.alpha,
            },
            'argument_model': {
                'name': self.am_name,
                'features': self.arg_features,
            },
            'stance_model': {
                'name': self.am_name,
                'features': self.stance_features,
            }
        }
        return json.dumps(d)

    @classmethod
    def from_json(cls, json_str) -> 'Configuration':
        d: dict = json.loads(json_str)
        rs = d.get('retrieval_system', dict())
        cls.topic_weight = rs.get('topic_weight', cls.topic_weight)
        cls.argument_weight = rs.get('argument_weight', cls.argument_weight)
        cls.prefetch_top_k = rs.get('prefetch_top_k', cls.prefetch_top_k)

        tm = d.get('topic_model', dict())
        cls.tm_name = tm.get('name', cls.tm_name)
        cls.tq_alpha = tm.get('tq_alpha', cls.tq_alpha)
        cls.alpha = tm.get('alpha', cls.alpha)

        am = d.get('argument_model', dict())
        cls.am_name = am.get('name', cls.am_name)
        cls.arg_features = am.get('features', cls.arg_features)

        sm = d.get('stance_model', dict())
        cls.sm_name = sm.get('name', cls.sm_name)
        cls.stance_features = sm.get('features', cls.stance_features)

        return cls

    def get_short(self) -> str:
        return f'rs:{self.topic_weight}-{self.argument_weight}-{self.prefetch_top_k}\n' + \
               f'tm:{self.alpha}-{self.tq_alpha}\n' \
               f'am:{self.arg_features}\n' \
               f'sm:{self.stance_features}'
