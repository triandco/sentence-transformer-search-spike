from typing_extensions import TypedDict

class TestCase(TypedDict):
  query: str
  expect: str

class RankResult(TypedDict):
  case: str
  result: 'list[tuple[str, float]]'