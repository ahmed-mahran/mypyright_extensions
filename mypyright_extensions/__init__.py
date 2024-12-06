from __future__ import annotations
import ast
from abc import ABC
from copy import copy
import functools
from types import ModuleType, NoneType
from typing import Annotated, Any, Callable, Protocol, Self, Type, cast


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
class TypeMap[*Params](ABC):
  @staticmethod
  def map_type(type_expr: str) -> str:
    raise NotImplementedError(type_expr)


class ITypeRefinementStatus(ABC):
  pass


class TypeRefinementStatus(ABC):
  class _Ok(ITypeRefinementStatus):
    def __repr__(self) -> str:
      return "Ok"

  Ok = _Ok()

  class Undecidable(ITypeRefinementStatus):
    message: str
    def __init__(self, message: str) -> None:
      self.message = message
      
    def __repr__(self) -> str:
      return 'Undecidable(' + self.message + ')'
    
  class Error(ITypeRefinementStatus):
    message: str
    def __init__(self, message: str) -> None:
      self.message = message
      
    def __repr__(self) -> str:
      return 'Error(' + self.message + ')'
    
  @staticmethod
  def is_ok(status: ITypeRefinementStatus | None) -> bool:
    return status is not None and isinstance(status, TypeRefinementStatus._Ok)


type TypeAST = ast.expr | ast.type_param


class RefinedType:
  # Type expression to be parsed
  _tp: TypeAST

  # Type attributes to be evaluated as python type, composed of
  # Python literal structures: strings, bytes, numbers,
  # tuples, lists, dicts, sets, booleans, None and Ellipsis
  # Those attributes are directly set and tested by predicates.
  _attributes: dict[str, Any]

  def __init__(self, tp: TypeAST, attributes: dict[str, Any] | None = None):
    self._tp = tp
    self._attributes = attributes if attributes is not None else {}

  @property
  def tp(self) -> TypeAST:
    return self._tp
  
  @property
  def attributes(self) -> dict[str, Any]:
    return copy(self._attributes)

  def set_from(self, other: RefinedType) -> RefinedType:
    clone = self.clone()

    for k, v in other._attributes.items():
      if isinstance(v, RefinedType):
        self_v = clone._attributes.get(k, None)
        clone._attributes[k] = v if self_v is None else self_v.set_from(v)
      else:
        clone._attributes[k] = v
    
    return clone
  
  def get_attribute(self, key: str) -> Any | None:
    return self._attributes.get(key)
  
  def set_attribute(self, key: str, value: Any) -> RefinedType:
    clone = self.clone()
    clone._attributes[key] = value
    return clone
  
  def update_attribute(self, key: str, update: Callable[[Any], Any], compute: Callable[[], Any]) -> RefinedType:
    clone = self.clone()
    clone._attributes[key] = update(clone._attributes[key]) if key in clone._attributes else compute()
    return clone
  
  def clone(self) -> RefinedType:
    return RefinedType(copy(self._tp), copy(self._attributes))
  
  @property
  def tp_repr(self) -> str:
    return ast.unparse(self._tp)
  
  def __repr__(self) -> str:
    return (
      self.tp_repr + ' ' +
      str(self._attributes)
    )


class TypeRefinementResult:
  refined_type: RefinedType
  status: ITypeRefinementStatus

  def __init__(self, refined_type: RefinedType, status: ITypeRefinementStatus):
    self.refined_type = refined_type
    self.status = status

  @staticmethod
  def is_ok(result: TypeRefinementResult | None) -> bool:
    return result is not None and isinstance(result.status, TypeRefinementStatus._Ok)
  
  @staticmethod
  def is_error(result: TypeRefinementResult | None) -> bool:
    return result is not None and isinstance(result.status, TypeRefinementStatus.Error)
  
  @staticmethod
  def is_undecidable(result: TypeRefinementResult | None) -> bool:
    return result is not None and isinstance(result.status, TypeRefinementStatus.Undecidable)
  
  def __str__(self) -> str:
    return str(self.refined_type) + ' ' + str(self.status)


class TypeRefinementPredicate[*Params]:
  @classmethod
  def init_type(cls, type_to_be_refined: RefinedType) -> RefinedType:
    return type_to_be_refined
  
  @classmethod
  def refine_type(
    cls,
    type_to_be_refined: RefinedType,
    args: list[Refinement],
    assume: bool,
  ) -> TypeRefinementResult:
    return TypeRefinementResult(type_to_be_refined, TypeRefinementStatus.Ok)

  @classmethod
  def type_repr(cls, args: list[RefinementFn[str]]) -> str:
    return cls.__name__


class This(TypeRefinementPredicate):
  @classmethod
  def refine_type(
    cls,
    type_to_be_refined: RefinedType,
    args: list[Refinement],
    assume: bool
  ) -> TypeRefinementResult:
    return TypeRefinementResult(type_to_be_refined, TypeRefinementStatus.Ok)


class TypeAsFunction[T, Result]:
  _origin: ast.expr
  _base: type[T] | ast.expr
  _args: list[TypeAsFunction[T, Result]]
  _result: Result | None

  def __init__(self, origin: ast.expr, base: type[T] | ast.expr, args: list[TypeAsFunction[T, Result]] | None = None, result: Result | None = None):
    self._origin = origin
    self._base = base
    self._args = args if args is not None else []
    self._result = result

  @property
  def origin(self) -> ast.expr:
    return self._origin
  
  @property
  def base(self) -> type[T] | ast.expr:
    return self._base

  @property
  def args(self) -> list[TypeAsFunction[T, Result]]:
    return self._args

  @property
  def result(self) -> Result | None:
    return self._result

  def __repr__(self):
    return (
      ((self._base.__name__ if hasattr(self._base, '__name__') else str(self._base)) if isinstance(self._base, type) else ast.unparse(self._base)) +
      ('[' + ', '.join([str(arg) for arg in self._args]) + ']' if len(self._args) > 0 else '') +
      (' => ' + str(self._result) if self._result is not None else '')
    )


def parse_str_dict(dict_expr: str) -> dict[str, str]:
  str_dict = cast(dict[str, str], ast.literal_eval(dict_expr))
  assert isinstance(str_dict, dict)
  return str_dict


class SymbolTable[T]:
  symbol_table: dict[str, str]
  reference_table: dict[str, type[T]]
  processed_files: set[str]
  module: ModuleType

  def __init__(self, symbol_table: str | dict[str, str], resolved: dict[str, type[T]] | None = None) -> None:
    if isinstance(symbol_table, str):
      symbol_table = dict([(k, v) for k, v in parse_str_dict(symbol_table).items() if v is not None])
    self.symbol_table = symbol_table
    self.reference_table = {**resolved} if resolved is not None else {}
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
    result_type: type[Result]
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

      if base_expr_id == 'Annotated':
        # it is a type of interest, continue traversal
        return TypeAsFunction(expr, base_expr, [_resolve_types(arg) for arg in args])
      
      base_expr_type = symbol_table.resolve_reference(base_expr_id)

      if base_expr_type is not None:
        # it is a type of interest, continue traversal
        return TypeAsFunction(expr, base_expr_type, [_resolve_types(arg) for arg in args])
      else:
        # not an interesting type, and we can terminate traversal of this path
        return TypeAsFunction(expr, expr, [])
    
    return _resolve_types(expr)

def parse_type_expr(type_expr: str) -> ast.expr:
  return cast(ast.Expr, ast.parse(type_expr).body[0]).value

def assume[T, P: TypeRefinementPredicate](instance: T, predicate: type[P]) -> Annotated[T, P]:
  return instance #type ignore

type RefinementFn[Result] = TypeAsFunction[TypeRefinementPredicate, Result]
type Refinement = RefinementFn[TypeRefinementResult]

def refine(
    tp: ast.expr,
    init_predicate: type[TypeRefinementPredicate] | None,
    tests: list[Refinement] | None,
    assumptions: list[Refinement] | None,
    typevar_bound_table: dict[str, Refinement | None],
    typevar_refinement_table: dict[str, Refinement],
    recursion_level: int = 0,
) -> Refinement:
  refined_type = RefinedType(tp)
  if isinstance(tp, ast.Name) and tp.id in typevar_bound_table:
    typevar_bound = typevar_bound_table[tp.id]
    if typevar_bound is not None and isinstance(typevar_bound.base, ast.Name) and typevar_bound.base.id == 'Annotated':
      refined_type = RefinedType(ast.TypeVar(name=tp.id, bound=typevar_bound.args[0].origin))
    else:
      bound_type = typevar_bound.origin if typevar_bound is not None else None
      refined_type = RefinedType(ast.TypeVar(name=tp.id, bound=bound_type))

  if init_predicate is not None:
    refined_type = init_predicate.init_type(refined_type)

  def _refine_all(predicates: list[Refinement], assume: bool):
    nonlocal refined_type

    tp = ast.Name(id=getattr(refined_type.tp, 'name')) if isinstance(refined_type.tp, ast.type_param) else refined_type.tp
    result: Refinement = TypeAsFunction(tp, tp, [], TypeRefinementResult(refined_type, TypeRefinementStatus.Ok))

    for predicate in predicates:
      if isinstance(predicate.base, type):
        refined_type = predicate.base.init_type(refined_type)
        result = _refine(predicate, refined_type, predicate.base, typevar_bound_table, typevar_refinement_table, assume, recursion_level)
        if TypeRefinementResult.is_ok(result.result) and result.result is not None:
          refined_type = result.result.refined_type
        else:
          return result
    
    return result

  tp = ast.Name(id=getattr(refined_type.tp, 'name')) if isinstance(refined_type.tp, ast.type_param) else refined_type.tp
  result: Refinement = TypeAsFunction(tp, tp, [], TypeRefinementResult(refined_type, TypeRefinementStatus.Ok))

  if assumptions is not None and len(assumptions) > 0:
    result = _refine_all(assumptions, assume=True)
    if result is None or not TypeRefinementResult.is_ok(result.result):
      return result

  if tests is not None and len(tests) > 0:
    result = _refine_all(tests, assume=True)

  print(''.join([' '] * recursion_level), 'result', result)

  return result

def _refine(
    refinement: Refinement,
    refined_type: RefinedType,
    init_predicate: type[TypeRefinementPredicate],
    typevar_bound_table: dict[str, Refinement | None],
    typevar_refinement_table: dict[str, Refinement],
    assume: bool,
    recursion_level: int = 0,
) -> Refinement:
  
  def _call_refine_assuming(
      refinement: Refinement,
      init_predicate: type[TypeRefinementPredicate] | None,
      assumptions: list[Refinement] | None,
      recursion_level: int = 0,
  ):
    return refine(
      refinement.origin,
      init_predicate=init_predicate,
      tests=None,
      assumptions=assumptions,
      typevar_bound_table=typevar_bound_table,
      typevar_refinement_table=typevar_refinement_table,
      recursion_level=recursion_level
    )
  
  def _traverse_refinement(
    refinement: Refinement,
    refined_type: RefinedType,
    init_predicate: type[TypeRefinementPredicate],
    recursion_level: int = 0,
  ) -> Refinement | None:
    print(''.join([' '] * recursion_level), '_traverse_refinement', refinement)
    if isinstance(refinement.base, ast.expr):

      match refinement.base:
        case ast.Name(id='Annotated'):
          return _call_refine_assuming(refinement.args[0], init_predicate=None, assumptions=refinement.args[1:])
        
        case ast.Name(id=typevar_name) if typevar_name in typevar_bound_table:
          typevar_refinement = typevar_refinement_table.get(typevar_name)
          if typevar_refinement is None or typevar_refinement.result is None:
            typevar_bound = typevar_bound_table[typevar_name]
            assumptions = None
            if typevar_bound is not None and isinstance(typevar_bound.base, ast.Name) and typevar_bound.base.id == 'Annotated':
              assumptions = typevar_bound.args[1:]
            typevar_refinement = _call_refine_assuming(refinement, init_predicate=init_predicate, assumptions=assumptions)
            typevar_refinement_table[typevar_name] = typevar_refinement
          return typevar_refinement

        case _: 
          return None
    
    else:

      args: list[Refinement] = []
      
      for arg in refinement.args:
        arg_result = _traverse_refinement(arg, refined_type, refinement.base, recursion_level + 1)
        args.append(arg_result if arg_result is not None else arg)

      result = refinement.base.refine_type(refined_type, args, assume)

      return TypeAsFunction(refinement.origin, refinement.base, args, result)
    
  result = _traverse_refinement(refinement, refined_type, init_predicate, recursion_level)

  if result is None or result.result is None:
    raise TypeError(f'Refinement result is None: {refinement}')
  
  return result

