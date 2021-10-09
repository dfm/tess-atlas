from abc import abstractmethod, ABC


class DataObject(ABC):

    @classmethod
    @abstractmethod
    def from_database(cls):
        pass

    @classmethod
    @abstractmethod
    def from_cache(cls):
        pass

    @abstractmethod
    def save_data(self):
        pass
