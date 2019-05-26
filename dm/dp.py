from dm.status import DMStatus


class Action:
    inform = 'inform'
    choose = 'choose'


class Policy:
    def __init__(self, action='', data=None, module=None):
        self.action = action
        self.data = data
        self.module = module

    def __str__(self):
        return str(self.__dict__)


class BaseDP:
    def __init__(self):
        pass

    def predict(self, nlu_result, status: DMStatus) -> Policy:
        raise NotImplementedError


class FirstDP(BaseDP):
    def predict(self, nlu_result, status: DMStatus) -> Policy:
        if nlu_result:
            r = nlu_result[0]
            module = r['module']
            r_type = r['type']
            if r_type == 'Brand':
                action = Action.choose
            else:
                action = Action.inform
            policy = Policy(action=action, data=r, module=module)
        else:
            policy = Policy()
        return policy
