"""Synthetic surface package.

The legacy re-export API was removed. Import directly from submodules:

- ``implied_volatility_diffusion.synthetic.heston``
- ``implied_volatility_diffusion.synthetic.sabr``
- ``implied_volatility_diffusion.synthetic.surface``
- ``implied_volatility_diffusion.synthetic.goals``
"""

__all__: list[str] = []


def __getattr__(name: str) -> object:
    raise AttributeError(
        f"`implied_volatility_diffusion.synthetic.{name}` no longer exists. "
        "Import from explicit submodules (e.g. "
        "`implied_volatility_diffusion.synthetic.heston`)."
    )
