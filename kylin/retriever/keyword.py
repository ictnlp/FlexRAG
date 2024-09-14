from dataclasses import dataclass


@dataclass
class Keyword:
    keyword: str
    weight: int = 1
    must: bool = False
    must_not: bool = False

    def to_dict(self):
        return {
            "keyword": self.keyword,
            "weight": self.weight,
            "must": self.must,
            "must_not": self.must_not,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            keyword=d["keyword"],
            weight=d["weight"],
            must=d["must"],
            must_not=d["must_not"],
        )


Keywords = list[Keyword]
