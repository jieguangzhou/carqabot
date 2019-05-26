from dm.status import UserStatusManager, DMStatus
from dm.dp import FirstDP, Policy


class DM:
    def __init__(self):
        self.user_status_manager = UserStatusManager()
        self.dp = FirstDP()

    def get_policy(self, nlu_result, history_status):
        policy = self.dp.predict(nlu_result, status=history_status)
        return policy

    def get_history_status(self, user_id) -> DMStatus:
        history_status = self.user_status_manager[user_id]
        return history_status

    def set_history_status(self):
        pass
