from collections import OrderedDict


# https://stackoverflow.com/questions/9917178/last-element-in-ordereddict
class MostRecentInsertedOrderedDict(OrderedDict):
    @property
    def last(self):
        if len(self) > 0:
            return next(reversed(self))
        return None

    @property
    def last_value(self):
        return self.get(self.last, None)
