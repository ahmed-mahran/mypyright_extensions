from __future__ import annotations
import ast
from abc import ABC
import functools
from types import ModuleType, NoneType
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
  
  def __call__(self, tp: Map[Type, *Ts] | Type[T], *args: P.args, **kwargs: P.kwargs) -> R:
    instance = self.instance
    owner = self.owner
    if instance is None and owner is None:
      fn = cast(_SubscriptableFunctionVariadic[*Ts, P, R] | _SubscriptableFunctionSingle[T, P, R], self.fn)
      # we depend on type checker to change tp type to match that of fn
      return fn(tp, *args, **kwargs) #type: ignore
    elif instance is not None and owner is not None:
      fn = cast(_SubscriptableMethodVariadic[Owner, *Ts, P, R] | _SubscriptableMethodSingle[Owner, T, P, R], self.fn)
      # we depend on type checker to change tp type to match that of fn
      return fn(instance, tp, *args, **kwargs) #type: ignore
    else:
      fn = cast(_SubscriptableClassMethodVariadic[Owner, *Ts, P, R] | _SubscriptableClassMethodSingle[Owner, T, P, R], self.fn)
      # we depend on type checker to change tp type to match that of fn
      return fn(owner, tp, *args, **kwargs) #type: ignore


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
  
  def __call__(self, tp: Map[Type, *Ts] | Type[T], *args: P.args, **kwargs: P.kwargs) -> R:
    # we depend on type checker to change tp type to match that of fn
    return self.fn(tp, *args, **kwargs) #type: ignore

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
  
  def __call__(self, tp: Map[Type, *Ts] | Type[T], *args: P.args, **kwargs: P.kwargs) -> R:
    # we depend on type checker to change tp type to match that of fn
    return self.fn(self.instance, tp, *args, **kwargs) #type: ignore

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
  
  def __call__(self, tp: Map[Type, *Ts] | Type[T], *args: P.args, **kwargs: P.kwargs) -> R:
    # we depend on type checker to change tp type to match that of fn
    return self.fn(self.owner, tp, *args, **kwargs) #type: ignore

#################################################################################################
def print_type(tp: type) -> str:
  def traverse_type(t: type) -> str:
    if hasattr(t, '__origin__'):
      repr = t.__origin__.__name__
      if len(t.__args__) > 0:
        repr += '[' + (',').join([traverse_type(arg) for arg in t.__args__]) + ']'
      return repr
    else:
      return t.__name__ if hasattr(t, '__name__') else str(t)
  return traverse_type(tp)

class TypeAsFunction[T, Result]:
  base: type[T] | ast.expr
  args: list[TypeAsFunction[T, Result]]
  result: Result | None = None

  def __init__(self, base: type[T] | ast.expr, args: list[TypeAsFunction[T, Result]] | None = None, result: Result | None = None):
    self.base = base
    self.args = args if args is not None else []
    self.result = result

  def __repr__(self):
    return (
      ((self.base.__name__ if hasattr(self.base, '__name__') else str(self.base)) if isinstance(self.base, type) else ast.dump(self.base)) +
      ('[' + ', '.join([str(arg) for arg in self.args]) + ']' if len(self.args) > 0 else '') +
      (' => ' + str(self.result) if self.result is not None else '')
    )

class SymbolTable[T]:
  #TODO we may add type variables definitions as well
  # key is type var name, value is type var definition
  symbol_table: dict[str, str]
  reference_table: dict[str, type[T]]
  processed_files: set[str]
  module: ModuleType

  def __init__(self, symbol_table: str | dict[str, str]) -> None:
    if isinstance(symbol_table, str):
      symbol_table = cast(dict[str, str], ast.literal_eval(symbol_table))
      assert isinstance(symbol_table, dict)
    self.symbol_table = symbol_table
    self.reference_table = {}
    self.processed_files = set()
    self.module = ModuleType('symbol_table_module')

  def resolve_reference(self, reference: str) -> type[T] | None:
    if reference in self.reference_table:
      return self.reference_table[reference]
    
    if reference in self.symbol_table:
      file = self.symbol_table[reference]
      if file not in self.processed_files:
        self.module.__dict__['__file__'] = file
        with open(file) as f:
          code = compile(f.read(), file, 'exec')
          exec(code, self.module.__dict__)
        self.processed_files.add(file)
      
      t = getattr(self.module, reference, None)
      if t is not None: # and is type
        self.reference_table[reference] = t
      return t

def resolve_types[T, Result](
    expr: ast.expr,
    symbol_table: SymbolTable[T],
    default_result: Result | None = None
  ) -> TypeAsFunction[T, Result]:
    def _resolve_types(expr: ast.expr) -> TypeAsFunction[T, Result]:
      args: list[ast.expr] = []

      base_expr = expr
      if isinstance(expr, ast.Subscript):
        base_expr = expr.value
        match expr.slice:
          case ast.Tuple(elts):
            args = elts
          case _:
            args = [expr.slice]

      base_expr_id = ast.unparse(base_expr)
      base_expr_type = symbol_table.resolve_reference(base_expr_id)

      if base_expr_type is not None:
        # it is a type of interest, continue traversal
        return TypeAsFunction(base_expr_type, [_resolve_types(arg) for arg in args], default_result)
      else:
        # not an interesting type, and we can terminate traversal of this path
        return TypeAsFunction(base_expr, [], default_result)
    
    return _resolve_types(expr)

def parse_type_expr(type_expr: str) -> ast.expr:
  return cast(ast.Expr, ast.parse(type_expr).body[0]).value

class TypeMap[*Params](ABC):
  @staticmethod
  def map_type(type_expr: str) -> str:
    raise NotImplementedError(type_expr)
