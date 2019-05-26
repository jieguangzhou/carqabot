class BaseStatus:
    @property
    def data(self):
        return self.__dict__

    def __str__(self):
        return str(self.data)


class KGStatus(BaseStatus):
    def __init__(self):
        self.status = []

    def add(self, entity=None, relation=None):
        entity = entity or self.history_entity
        relation = relation or self.history_relation
        status = {
            'entity': entity,
            'relation': relation
        }
        self.status.append(status)

    @property
    def last_entity(self):
        if self.status:
            entity = self.status[-1].get('entity')
        else:
            entity = None
        return entity

    @property
    def last_relation(self):
        if self.status:
            relation = self.status[-1].get('relation')
        else:
            relation = None
        return relation


class DPStatus(BaseStatus):
    def __init__(self):
        self.status = []

    def add(self, policy):
        self.status.append(policy)

    @property
    def last_policy(self):
        if self.status:
            policy = self.status[-1]
        else:
            policy = None
        return policy


class DMStatus(BaseStatus):
    def __init__(self):
        self.kb_status = KGStatus()
        self.dp_status = DPStatus()


class UserStatusManager:
    def __init__(self):
        self.user_status = {}

    def __getitem__(self, item):
        if item not in self.user_status:
            self.user_status[item] = DMStatus()
        return self.user_status[item]

    def __setitem__(self, key, value):
        if isinstance(value, DMStatus):
            self.user_status[key] = value
        else:
            raise Exception('value must be a DMStatus, but get {}'.format(type(value)))
