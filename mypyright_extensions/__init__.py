from abc import ABC
import functools
from types import NoneType
from typing import Callable, Protocol, Self, Type, cast


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

SUBSCRIPTABLE_WRAPPER_ASSIGNMENTS = functools.WRAPPER_ASSIGNMENTS + ('__code__', '__func__')

class _SubscriptableFunctionSingle[T, **P, R](Protocol):
  def __call__(self, tp: Type[T], /, *args: P.args, **kwargs: P.kwargs) -> R: ...

class _SubscriptableFunctionVariadic[*Ts, **P, R](Protocol):
  def __call__(self, tp: Map[Type, *Ts], /,  *args: P.args, **kwargs: P.kwargs) -> R: ...

class _SubscriptableMethodSingle[Owner, T, **P, R](Protocol):
  def __call__(self, instance: Owner, tp: Type[T], /, *args: P.args, **kwargs: P.kwargs) -> R: ...

class _SubscriptableMethodVariadic[Owner, *Ts, **P, R](Protocol):
  def __call__(self, instance: Owner, tp: Map[Type, *Ts], /, *args: P.args, **kwargs: P.kwargs) -> R: ...

class _SubscriptableClassMethodSingle[Owner, T, **P, R](Protocol):
  def __call__(self, owner: Type[Owner], tp: Type[T], /, *args: P.args, **kwargs: P.kwargs) -> R: ...

class _SubscriptableClassMethodVariadic[Owner, *Ts, **P, R](Protocol):
  def __call__(self, owner: Type[Owner], tp: Map[Type, *Ts], /, *args: P.args, **kwargs: P.kwargs) -> R: ...

class subscriptable[Owner, T, *Ts, **P, R]:
  def __init__(
      self,
      fn: (
        _SubscriptableFunctionVariadic[*Ts, P, R] |
        _SubscriptableFunctionSingle[T, P, R] |
        _SubscriptableMethodVariadic[Owner, *Ts, P, R] |
        _SubscriptableMethodSingle[Owner, T, P, R] |
        _SubscriptableClassMethodVariadic[Owner, *Ts, P, R] |
        _SubscriptableClassMethodSingle[Owner, T, P, R]
      )
  ) -> None:
    __func__ = getattr(fn, '__func__', fn)
    functools.update_wrapper(wrapper=self, wrapped=__func__, assigned=SUBSCRIPTABLE_WRAPPER_ASSIGNMENTS)
    self.fn = fn
    self.instance = None
    self.owner = None

  def __get__(self, instance: Owner | None, owner: Type[Owner]) -> Self:
    self.instance = instance
    self.owner = owner
    return self

  def __getitem__(self, tp: Map[Type, *Ts] | Type[T]) -> Callable[P, R]:
    instance = self.instance
    owner = self.owner
    if instance is None and owner is None:
      fn = cast(_SubscriptableFunctionVariadic[*Ts, P, R] | _SubscriptableFunctionSingle[T, P, R], self.fn)
      @functools.wraps(fn)
      def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        # we depend on type checker to change tp type to match that of fn
        return fn(tp, *args, **kwargs) #type: ignore
      return inner
    elif instance is not None and owner is not None:
      fn = cast(_SubscriptableMethodVariadic[Owner, *Ts, P, R] | _SubscriptableMethodSingle[Owner, T, P, R], self.fn)
      @functools.wraps(fn)
      def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        # we depend on type checker to change tp type to match that of fn
        return fn(instance, tp, *args, **kwargs) #type: ignore
      return inner
    else:
      fn = cast(_SubscriptableClassMethodVariadic[Owner, *Ts, P, R] | _SubscriptableClassMethodSingle[Owner, T, P, R], self.fn)
      @functools.wraps(fn)
      def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        # we depend on type checker to change tp type to match that of fn
        return fn(owner, tp, *args, **kwargs) #type: ignore
      return inner

    # # inner.__type_params__ = (T,)
    # return inner_function if self.instance is None else inner_method


class subscriptablefunction[T, *Ts, **P, R]:
  def __init__(self, fn: _SubscriptableFunctionVariadic[*Ts, P, R] | _SubscriptableFunctionSingle[T, P, R]) -> None:
    __func__ = getattr(fn, '__func__', fn)
    functools.update_wrapper(wrapper=self, wrapped=__func__, assigned=SUBSCRIPTABLE_WRAPPER_ASSIGNMENTS)
    self.fn = fn

  def __getitem__(self, tp: Map[Type, *Ts] | Type[T]) -> Callable[P, R]:
    @functools.wraps(self.fn)
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
      # we depend on type checker to change tp type to match that of fn
      return self.fn(tp, *args, **kwargs) #type: ignore
    return inner

class subscriptablemethod[Owner, T, *Ts, **P, R]:
  def __init__(self, fn: _SubscriptableMethodVariadic[Owner, *Ts, P, R] | _SubscriptableMethodSingle[Owner, T, P, R]) -> None:
    __func__ = getattr(fn, '__func__', fn)
    functools.update_wrapper(wrapper=self, wrapped=__func__, assigned=SUBSCRIPTABLE_WRAPPER_ASSIGNMENTS)
    self.fn = fn

  def __get__(self, instance: Owner, owner: Type[Owner]) -> Self:
    self.instance = instance
    return self

  def __getitem__(self, tp: Map[Type, *Ts] | Type[T]) -> Callable[P, R]:
    @functools.wraps(self.fn)
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
      # we depend on type checker to change tp type to match that of fn
      return self.fn(self.instance, tp, *args, **kwargs) #type: ignore
    # inner.__type_params__ = (T,)
    return inner

class subscriptableclassmethod[Owner, T, *Ts, **P, R]:
  def __init__(self, fn: _SubscriptableClassMethodVariadic[Owner, *Ts, P, R] | _SubscriptableClassMethodSingle[Owner, T, P, R]) -> None:
    __func__ = getattr(fn, '__func__', fn)
    functools.update_wrapper(wrapper=self, wrapped=__func__, assigned=SUBSCRIPTABLE_WRAPPER_ASSIGNMENTS)
    self.fn = fn

  def __get__(self, instance: NoneType, owner: Type[Owner]) -> Self:
    self.owner = owner
    return self

  def __getitem__(self, tp: Map[Type, *Ts] | Type[T]) -> Callable[P, R]:
    @functools.wraps(self.fn)
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
      # we depend on type checker to change tp type to match that of fn
      return self.fn(self.owner, tp, *args, **kwargs) #type: ignore
    # inner.__type_params__ = (T,)
    return inner
