#### Events

class Event(object):
    def __init__(self, i, original):
        self.i = i
        self.original = original

    def __eq__(self, other):
        return self.original == other.original and self.i == other.i

    @staticmethod
    def not_found(): raise Exception("event not found in vocab")


class PitchDurationEvent(Event):
    def __init__(self, i, pitch, duration):
        self.i = i
        self.pitch = pitch
        self.duration = duration

    def __eq__(self, other):
        return self.i 

    @staticmethod
    def not_found(): raise Exception("event not found in vocab")

