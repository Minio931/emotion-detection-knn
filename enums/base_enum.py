from enum import Enum

class BaseEnum(Enum):
    @classmethod
    def get_enum_value(cls, value) -> str:
        try:
            return cls(value).name.capitalize()
        except ValueError:
            return 'Unknown'