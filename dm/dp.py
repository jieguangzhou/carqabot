from dm.status import DMStatus


class Action:
    inform = 'inform'
    choose = 'choose'
    guide = 'guide'


class Policy:
    def __init__(self, action='', data=None, module=None):
        self.action = action
        self.data = data
        self.module = module

    def __str__(self):
        return str(self.__dict__)

    def __bool__(self):
        return not self.action == ''


class BaseDP:
    def __init__(self):
        pass

    def predict(self, nlu_result, status: DMStatus) -> Policy:
        raise NotImplementedError


class FirstDP(BaseDP):
    def predict(self, nlu_result, status: DMStatus) -> Policy:
        if nlu_result:
            module = nlu_result['module']
            if module == 'dbqa':
                return Policy(action=Action.inform, data=nlu_result, module=module)
            r_type = nlu_result['type']
            if r_type == 'Brand':
                action = Action.choose
            else:
                action = Action.inform
            policy = Policy(action=action, data=nlu_result, module=module)
        else:
            policy = Policy()
        return policy


class SampleDP(BaseDP):
    def predict(self, nlu_result, status: DMStatus) -> Policy:
        policy = Policy(action=Action.guide)
        return policy
