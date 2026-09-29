"""Router registry. Routers are built from the ``routers:`` section of a config:

    routers:
      routellm_bert: {type: routellm, args: {type: bert}}

``type`` selects the adapter class below and ``args`` are passed to it.
"""

from r2a.routers.base import HFEncoder, Router

ROUTER_TYPES = {
    "routellm": ("r2a.routers.routellm", "RouteLLMRouter"),
    "p2l": ("r2a.routers.p2l", "P2LRouter"),
    "graphrouter": ("r2a.routers.graphrouter", "GraphRouter"),
    "routerdc": ("r2a.routers.routerdc", "RouterDC"),
}


def create_router(router_type: str, **kwargs) -> Router:
    import importlib

    if router_type not in ROUTER_TYPES:
        raise ValueError(f"Unknown router type '{router_type}'. Available: {sorted(ROUTER_TYPES)}")
    module_name, class_name = ROUTER_TYPES[router_type]
    return getattr(importlib.import_module(module_name), class_name)(**kwargs)


__all__ = ["HFEncoder", "Router", "ROUTER_TYPES", "create_router"]
