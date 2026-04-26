"""Synthetic surface namespace package."""

__all__: list[str] = []


def __getattr__(name: str) -> object:
    raise AttributeError(
        f"`implied_volatility_diffusion.synthetic.{name}` no longer exists. "
        "Import from explicit submodules (e.g. "
        "`implied_volatility_diffusion.synthetic.heston`)."
    )
