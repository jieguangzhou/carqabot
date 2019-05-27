from dm.status import UserStatusManager, DMStatus
from dm.dp import FirstDP, Policy, SampleDP


class DM:
    def __init__(self):
        self.user_status_manager = UserStatusManager()
        self.dp_first = FirstDP()
        self.dp_sample = SampleDP()

    def get_policy(self, nlu_result, history_status):
        policy = self.dp_first.predict(nlu_result, status=history_status)
        if not policy:
            policy = self.dp_sample.predict(nlu_result, status=history_status)
        return policy

    def get_history_status(self, user_id) -> DMStatus:
        history_status = self.user_status_manager[user_id]
        return history_status

    def add_nlg_status(self, user_id, nlg_result):
        history_status = self.get_history_status(user_id)
        history_status.nlg_status.add(nlg_result)

    def add_nlu_status(self, user_id, kbqa_status):
        history_status = self.get_history_status(user_id)
        history_status.kb_status.add(**kbqa_status)
