# src/orchestrators/bootstrap.py
from __future__ import annotations

from configparser import ConfigParser
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Mapping, Type, TypeVar, Sequence

"""
Bootstrap générique pour orchestrateurs (style microservice interne).

Ordre de résolution des paramètres :
  1) Contexte explicite (ex. ConfigOrchestrator) -> plus prioritaire.
  2) Fichier INI dédié (ex. conf/orchestrators/<name>.ini).
  3) Defaults internes minimaux (constants codés en dur).

L'appelant fournit:
  - cls ou class_path (import dynamique) ou factory(params)->instance,
  - defaults, validator(instance)->None (DOIT lever si invalide),
  - context_provider(name)->mapping|None,
  - ini_filenames (ex. ['config.ini','default.ini']) et ini_dirs si besoin.
"""

T = TypeVar("T")

DEFAULT_INI_DIRS = (
    Path("./conf/orchestrators"),
    Path("./conf"),
    Path("/app/conf/orchestrators"),
    Path("/app/conf"),
)
DEFAULT_SECTION_PREFIX = "orchestrator:"

Factory = Callable[[Mapping[str, Any]], T]
Validator = Callable[[T], None]
ContextProvider = Callable[[str], Mapping[str, Any] | None]


def _read_ini(path: Path, section: str) -> dict[str, Any]:
    if not path.is_file():
        return {}
    cp = ConfigParser()
    cp.read(path)
    if not cp.has_section(section):
        return {}
    return {k: v for k, v in cp.items(section)}


def resolve_class(class_path: str) -> Type[Any]:
    module_path, _, cls_name = class_path.replace(":", ".").rpartition(".")
    if not module_path or not cls_name:
        raise ValueError(f"resolve_class: chemin invalide '{class_path}'")
    mod = import_module(module_path)
    cls = getattr(mod, cls_name, None)
    if cls is None:
        raise ImportError(f"resolve_class: classe '{cls_name}' introuvable dans '{module_path}'")
    return cls  # type: ignore[return-value]


def resolve_params(
    *,
    name: str,
    defaults: Mapping[str, Any],
    context: Mapping[str, Any] | None,
    ini_filenames: Sequence[str] | None = None,
    ini_dirs: Sequence[Path] = DEFAULT_INI_DIRS,
    section_prefix: str = DEFAULT_SECTION_PREFIX,
) -> dict[str, Any]:
    params: dict[str, Any] = dict(defaults)
    if context:
        params.update({k: v for k, v in context.items() if v not in (None, "")})
    filenames = list(ini_filenames or (f"{name}.ini", "default.ini"))
    section = f"{section_prefix}{name}"
    merged_from_file: dict[str, Any] = {}
    for d in ini_dirs:
        for fn in filenames:
            candidate = (d / fn).resolve()
            got = _read_ini(candidate, section)
            if got:
                merged_from_file = got
                break
        if merged_from_file:
            break
    if merged_from_file:
        params.update(merged_from_file)
    return params


def bootstrap_instance(
    *,
    name: str,
    cls: Type[T] | None = None,
    class_path: str | None = None,
    factory: Factory | None = None,
    defaults: Mapping[str, Any],
    validator: Validator,
    context_provider: ContextProvider | None = None,
    ini_filenames: Sequence[str] | None = None,
    ini_dirs: Sequence[Path] = DEFAULT_INI_DIRS,
    section_prefix: str = DEFAULT_SECTION_PREFIX,
) -> T:
    if sum(x is not None for x in (cls, class_path, factory)) != 1:
        raise ValueError("bootstrap_instance: fournir exactement l'un de 'cls', 'class_path' ou 'factory'")
    context = context_provider(name) if context_provider else None
    params = resolve_params(
        name=name,
        defaults=defaults,
        context=context,
        ini_filenames=ini_filenames,
        ini_dirs=ini_dirs,
        section_prefix=section_prefix,
    )
    instance: T
    if factory is not None:
        instance = factory(params)
    else:
        the_cls: Type[Any] = cls or resolve_class(class_path or "")
        kwargs = {str(k): v for k, v in params.items()}
        instance = the_cls(**kwargs)  # type: ignore[misc]
    validator(instance)
    return instance
