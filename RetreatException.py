
class RetreatException(Exception):

    def __init__(self, initdate,targetdate,message="Retreat"):
        self.initdate = initdate
        self.targetdate = targetdate
        super().__init__(message)