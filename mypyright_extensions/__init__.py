from abc import ABC
from typing import Callable, Concatenate, Type


class _MyPyright(ABC):
  pass

class Map[F, *Ts](_MyPyright):
  """
  Map[type, int]          ==== type[int]
  Map[type, int, str]     ==== *tuple[type[int], type[str]]
  Map[type, T]            ==== type[T]
  Map[type, T: int]       ==== type[T: int]
  Map[type, *Ts]          ==== *Ts: type
  Map[type, int, T, *Ts]  ==== *tuple[type[int], type[T], *Ts: type]
  """
  pass

class subscriptable[*T, **P, R]:
  def __init__(self, fn: Callable[Concatenate[Map[Type, *T], P], R]) -> None:
    self.fn = fn

  def __getitem__(self, tp: Map[Type, *T]) -> Callable[P, R]:
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
      return self.fn(tp, *args, **kwargs)
    # inner.__type_params__ = (T,)
    return inner
