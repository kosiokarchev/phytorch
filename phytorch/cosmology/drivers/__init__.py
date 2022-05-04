from types import ModuleType


class _FLRWDriverModule(ModuleType):
    from typing import Type

    from .. import special

    LambdaCDM: Type[special.LambdaCDM]
    LambdaCDMR: Type[special.LambdaCDMR]
    wCDM: Type[special.wCDM]
    wCDMR: Type[special.wCDMR]
    w0waCDM: Type[special.w0waCDM]
    w0waCDMR: Type[special.w0waCDMR]
    wpwaCDM: Type[special.wpwaCDM]
    wpwaCDMR: Type[special.wpwaCDMR]
    w0wzCDM: Type[special.w0wzCDM]
    w0wzCDMR: Type[special.w0wzCDMR]
    FlatLambdaCDM: Type[special.FlatLambdaCDM]
    FlatLambdaCDMR: Type[special.FlatLambdaCDMR]
    FlatwCDM: Type[special.FlatwCDM]
    FlatwCDMR: Type[special.FlatwCDMR]
    Flatw0waCDM: Type[special.Flatw0waCDM]
    Flatw0waCDMR: Type[special.Flatw0waCDMR]
    FlatwpwaCDM: Type[special.FlatwpwaCDM]
    FlatwpwaCDMR: Type[special.FlatwpwaCDMR]
    Flatw0wzCDM: Type[special.Flatw0wzCDM]
    Flatw0wzCDMR: Type[special.Flatw0wzCDMR]


analytic: _FLRWDriverModule
analytic_diff: _FLRWDriverModule
odeint: _FLRWDriverModule
