# Based on the astropy test suite (v4.2.1)
# (https://github.com/astropy/astropy/blob/v4.2.1/astropy/cosmology/tests/test_cosmology.py)
import re
from io import StringIO
from typing import Type

import numpy as np
import pytest
import torch
from pytest import mark
from torch import tensor

import phytorch.cosmology.drivers.abstract as abstract
import phytorch.cosmology.drivers.analytic as analytic
import phytorch.cosmology.drivers.analytic_diff as analytic_diff
import phytorch.cosmology.drivers.odeint as odeint
import phytorch.cosmology.special as special
from phytorch.constants import codata2014, G as Newton_G
from phytorch.units import Unit
from phytorch.units.astro import Gpc, Gyr, Mpc
from phytorch.units.si import cm, gram, kelvin, km, s

from tests.common.closeness import close
from tests.common.dtypes import with_default_double


ZERO = torch.zeros(())
ONE = torch.ones(())
SMALL = 1e-16
Z = tensor([0, 0.5, 1, 2])

H70 = 70 * km/s/Mpc
H704 = 70.4 * km/s/Mpc


def test_critical_density():
    fac = (Newton_G / codata2014.G).to(Unit())

    cosmo = abstract.FlatLambdaCDMR(H0=H704, Om0=0.272)

    # constants defined only so accurately
    assert ((cosmo.critical_density0 * fac).to(gram / cm**3) - 9.309668456020899e-30) < 1e-9
    assert cosmo.critical_density0 == cosmo.critical_density(0)

    assert close((cosmo.critical_density(tensor([1, 5])) * fac).to(gram / cm**3).value,
                 [2.70352772e-29, 5.53739080e-28])


def test_xtfuncs():
    cosmo = abstract.LambdaCDMR(
        H0=H70, Om0=0.3, Ode0=0.5, Neff=3.04, Tcmb0=2.725 * kelvin
    )

    z = tensor([2, 3.2])
    assert close(cosmo.lookback_time_integrand(tensor(3)), 0.052218976654969378)
    assert close(cosmo.lookback_time_integrand(z), [0.10333179, 0.04644541])
    assert close(cosmo.abs_distance_integrand(tensor(3)), 3.3420145059180402)
    assert close(cosmo.abs_distance_integrand(z), [2.7899584, 3.44104758])


def test_zeroing():
    cosmo = abstract.LambdaCDMR(Om0=0.27, Ode0=0, Or0=0)

    assert cosmo.Ode(1.5) == 0
    assert (cosmo.Ode(Z) == ZERO).all()
    assert cosmo.Or(1.5) == 0
    assert (cosmo.Or(Z) == ZERO).all()
    # TODO: add neutrinos
    # assert allclose(cosmo.Onu(1.5), [0, 0, 0, 0])
    # assert allclose(cosmo.Onu(z), [0, 0, 0, 0])
    assert (cosmo.Ob(Z) == ZERO).all()


def test_matter():
    cosmo = abstract.FlatLambdaCDMR(Om0=0.3, Ob0=0.045)

    assert cosmo.Om(0) == 0.3
    assert cosmo.Ob(0) == 0.045
    assert close(cosmo.Om(Z), [0.3, 0.59124088, 0.77419355, 0.92045455])
    assert close(cosmo.Ob(Z), [0.045, 0.08868613, 0.11612903, 0.13806818])
    assert close(cosmo.Odm(Z), [0.255, 0.50255474, 0.65806452, 0.78238636])
    assert close(cosmo.Ob(Z) + cosmo.Odm(Z), cosmo.Om(Z))


def test_ocurv():
    cosmo = abstract.FlatLambdaCDMR(Om0=0.3)

    assert cosmo.Ok0 == 0
    assert cosmo.Ok(0) == 0
    assert (cosmo.Ok(Z) == ZERO).all()

    cosmo = abstract.LambdaCDMR(Om0=0.3, Ode0=0.5)
    assert abs(cosmo.Ok0 - 0.2) < SMALL
    assert abs(cosmo.Ok(0) - 0.2) < SMALL
    assert close(cosmo.Ok(Z), [0.2, 0.22929936, 0.21621622, 0.17307692])

    assert (cosmo.Ok(Z) + cosmo.Om(Z) + cosmo.Ode(Z) == ONE).all()


def test_ode():
    cosmo = abstract.FlatLambdaCDMR(Om0=0.3)

    assert cosmo.Ode(0) == cosmo.Ode0
    assert close(cosmo.Ode(Z), [0.7, 0.408759, 0.2258065, 0.07954545])


def test_tcmb():
    cosmo = abstract.FlatLambdaCDMR(H0=H704, Om0=0.272, Tcmb0=2.5*kelvin)

    assert cosmo.Tcmb(2) == 7.5 * kelvin
    assert (cosmo.Tcmb(tensor([0, 1, 2, 3, 9.])).to(kelvin).value == tensor([2.5, 5, 7.5, 10, 25])).all()


def test_efunc_vs_invefunc():
    cosmo = abstract.LambdaCDMR(Om0=0.3, Ode0=0.7)

    assert cosmo.efunc(0.5) * cosmo.inv_efunc(0.5) == 1
    assert (cosmo.efunc(Z) * cosmo.inv_efunc(Z) == ONE).all()
    # TODO: test this for subclasses?


class BaseLambdaCDMDriverTest:
    flat_cosmo_cls: Type[special.FlatLambdaCDM]
    cosmo_cls: Type[special.LambdaCDM]


class BaseLambdaCDMTest(BaseLambdaCDMDriverTest):
    flat_cosmo_cls: Type[special.FlatLambdaCDM]
    cosmo_cls: Type[special.LambdaCDM]

    @with_default_double
    @mark.parametrize(('func', 'vals', 'unit', 'rtol'), (
        # From the astropy test suite:
        # Test values were taken from the following web cosmology
        # calculators on 27th Feb 2012:
        # Wright: http://www.astro.ucla.edu/~wright/CosmoCalc.html
        #         (https://ui.adsabs.harvard.edu/abs/2006PASP..118.1711W)
        # Kempner: http://www.kempner.net/cosmic.php
        # iCosmos: http://www.icosmos.co.uk/index.html
        (special.FlatLambdaCDM.comoving_distance,
         (3364.5, 3364.8, 3364.7988), Mpc, 1e-4),
        (special.FlatLambdaCDM.angular_diameter_distance,
         (1682.3, 1682.4, 1682.3994), Mpc, 1e-4),
        (special.FlatLambdaCDM.luminosity_distance,
         (6729.2, 6729.6, 6729.5976), Mpc, 1e-4),
        (special.FlatLambdaCDM.lookback_time,
         (7.841, 7.84178, 7.843), Gyr, 1e-3),
        (special.FlatLambdaCDM.lookback_distance,
         (2404.0, 2404.24, 2404.4), Mpc, 1e-3),
    ))
    def test_flat_z1(self, func, vals, unit, rtol):
        assert close(getattr(self.flat_cosmo_cls(H0=H70, Om0=0.27), func.__name__)(1).to(unit).value, vals, rtol=rtol)

    @mark.parametrize('Om0, Ode0, vals', (
        (0.27, 0.73, (29.123, 159.529, 630.427, 1178.531, 2181.485, 3654.802)),
        (0.27, 0, (20.501, 99.019, 380.278, 747.049, 1558.363, 3123.814)),
        (2, 0, (12.619, 44.708, 114.904, 173.709, 258.82, 358.992))
    ))
    def test_comoving_volume(self, Om0, Ode0, vals):
        z = tensor([0.5, 1, 2, 3, 5, 9])
        # for (Om0, Ode0), vals in zip(
        #     ((0.27, 0.73), (0.27, 0), (2, 0)),
        #     # Form Ned Wright's calculator: not very *accurate* (sic), so
        #     # like astropy, test to very low precision
        #     ((29.123, 159.529, 630.427, 1178.531, 2181.485, 3654.802),
        #      (20.501, 99.019, 380.278, 747.049, 1558.363, 3123.814),
        #      (12.619, 44.708, 114.904, 173.709, 258.82, 358.992))
        # ):
        assert close(self.cosmo_cls(H0=H70, Om0=Om0, Ode0=Ode0).comoving_volume(z).to(Gpc**3).value, vals, rtol=1e-2)

    # TODO: (requires integration) test_differential_comoving_volume

    icosmo_flat = """\
    # from icosmo (icosmo.org)
    # Om 0.3 w -1 h 0.7 Ol 0.7
    # z     comoving_transvers_dist   angular_diameter_dist  luminosity_dist
           0.0000000       0.0000000       0.0000000         0.0000000
          0.16250000       669.77536       576.15085         778.61386
          0.32500000       1285.5964       970.26143         1703.4152
          0.50000000       1888.6254       1259.0836         2832.9381
          0.66250000       2395.5489       1440.9317         3982.6000
          0.82500000       2855.5732       1564.6976         5211.4210
           1.0000000       3303.8288       1651.9144         6607.6577
           1.1625000       3681.1867       1702.2829         7960.5663
           1.3250000       4025.5229       1731.4077         9359.3408
           1.5000000       4363.8558       1745.5423         10909.640
           1.6625000       4651.4830       1747.0359         12384.573
           1.8250000       4916.5970       1740.3883         13889.387
           2.0000000       5179.8621       1726.6207         15539.586
           2.1625000       5406.0204       1709.4136         17096.540
           2.3250000       5616.5075       1689.1752         18674.888
           2.5000000       5827.5418       1665.0120         20396.396
           2.6625000       6010.4886       1641.0890         22013.414
           2.8250000       6182.1688       1616.2533         23646.796
           3.0000000       6355.6855       1588.9214         25422.742
           3.1625000       6507.2491       1563.3031         27086.425
           3.3250000       6650.4520       1537.6768         28763.205
           3.5000000       6796.1499       1510.2555         30582.674
           3.6625000       6924.2096       1485.0852         32284.127
           3.8250000       7045.8876       1460.2876         33996.408
           4.0000000       7170.3664       1434.0733         35851.832
           4.1625000       7280.3423       1410.2358         37584.767
           4.3250000       7385.3277       1386.9160         39326.870
           4.5000000       7493.2222       1362.4040         41212.722
           4.6625000       7588.9589       1340.2135         42972.480
    """

    icosmo_open = """\
    # from icosmo (icosmo.org)
    # Om 0.3 w -1 h 0.7 Ol 0.1
    # z     comoving_transvers_dist   angular_diameter_dist  luminosity_dist
           0.0000000       0.0000000       0.0000000       0.0000000
          0.16250000       643.08185       553.18868       747.58265
          0.32500000       1200.9858       906.40441       1591.3062
          0.50000000       1731.6262       1154.4175       2597.4393
          0.66250000       2174.3252       1307.8648       3614.8157
          0.82500000       2578.7616       1413.0201       4706.2399
           1.0000000       2979.3460       1489.6730       5958.6920
           1.1625000       3324.2002       1537.2024       7188.5829
           1.3250000       3646.8432       1568.5347       8478.9104
           1.5000000       3972.8407       1589.1363       9932.1017
           1.6625000       4258.1131       1599.2913       11337.226
           1.8250000       4528.5346       1603.0211       12793.110
           2.0000000       4804.9314       1601.6438       14414.794
           2.1625000       5049.2007       1596.5852       15968.097
           2.3250000       5282.6693       1588.7727       17564.875
           2.5000000       5523.0914       1578.0261       19330.820
           2.6625000       5736.9813       1566.4113       21011.694
           2.8250000       5942.5803       1553.6158       22730.370
           3.0000000       6155.4289       1538.8572       24621.716
           3.1625000       6345.6997       1524.4924       26413.975
           3.3250000       6529.3655       1509.6799       28239.506
           3.5000000       6720.2676       1493.3928       30241.204
           3.6625000       6891.5474       1478.0799       32131.840
           3.8250000       7057.4213       1462.6780       34052.058
           4.0000000       7230.3723       1446.0745       36151.862
           4.1625000       7385.9998       1430.7021       38130.224
           4.3250000       7537.1112       1415.4199       40135.117
           4.5000000       7695.0718       1399.1040       42322.895
           4.6625000       7837.5510       1384.1150       44380.133
    """

    icosmo_closed = """\
    # from icosmo (icosmo.org)
    # Om 2 w -1 h 0.7 Ol 0.1
    # z     comoving_transvers_dist   angular_diameter_dist  luminosity_dist
           0.0000000       0.0000000       0.0000000       0.0000000
          0.16250000       601.80160       517.67879       699.59436
          0.32500000       1057.9502       798.45297       1401.7840
          0.50000000       1438.2161       958.81076       2157.3242
          0.66250000       1718.6778       1033.7912       2857.3019
          0.82500000       1948.2400       1067.5288       3555.5381
           1.0000000       2152.7954       1076.3977       4305.5908
           1.1625000       2312.3427       1069.2914       5000.4410
           1.3250000       2448.9755       1053.3228       5693.8681
           1.5000000       2575.6795       1030.2718       6439.1988
           1.6625000       2677.9671       1005.8092       7130.0873
           1.8250000       2768.1157       979.86398       7819.9270
           2.0000000       2853.9222       951.30739       8561.7665
           2.1625000       2924.8116       924.84161       9249.7167
           2.3250000       2988.5333       898.80701       9936.8732
           2.5000000       3050.3065       871.51614       10676.073
           2.6625000       3102.1909       847.01459       11361.774
           2.8250000       3149.5043       823.39982       12046.854
           3.0000000       3195.9966       798.99915       12783.986
           3.1625000       3235.5334       777.30533       13467.908
           3.3250000       3271.9832       756.52790       14151.327
           3.5000000       3308.1758       735.15017       14886.791
           3.6625000       3339.2521       716.19347       15569.263
           3.8250000       3368.1489       698.06195       16251.319
           4.0000000       3397.0803       679.41605       16985.401
           4.1625000       3422.1142       662.87926       17666.664
           4.3250000       3445.5542       647.05243       18347.576
           4.5000000       3469.1805       630.76008       19080.493
           4.6625000       3489.7534       616.29199       19760.729
    """

    @mark.parametrize('Om0, Ode0, data', (
        (0.3, 0.7, icosmo_flat), (0.3, 0.1, icosmo_open), (2, 0.1, icosmo_closed)
    ))
    def test_flat_open_closed_icosmo(self, Om0, Ode0, data):
        cosmo = self.cosmo_cls(H0=H70, Om0=Om0, Ode0=Ode0)

        z, dm, da, dl = (tensor(_, dtype=torch.get_default_dtype())
                         for _ in np.loadtxt(StringIO(data), unpack=True))

        assert close(cosmo.comoving_transverse_distance(z).to(Mpc).value, dm)
        assert close(cosmo.angular_diameter_distance(z).to(Mpc).value, da)
        assert close(cosmo.luminosity_distance(z).to(Mpc).value, dl)

    def test_distmod(self):
        cosmo = self.flat_cosmo_cls(H0=H704, Om0=0.272)

        assert cosmo.hubble_distance.to(Mpc) == 4258.415596590909
        assert close(cosmo.distmod(tensor([1, 5])), [44.124857, 48.40167258])

    @with_default_double
    def test_negdistmod(self):
        cosmo = self.cosmo_cls(H0=H70, Om0=0.2, Ode0=1.3)
        z = tensor([50, 100])
        assert close(cosmo.luminosity_distance(z).to(Mpc).value, [16612.44047622, -46890.79092244])
        assert close(cosmo.distmod(z), [46.102167189, 48.355437790944])

    def test_comoving_distance_z1z2(self):
        cosmo = self.cosmo_cls(Om0=0.3, Ode0=0.8)

        with pytest.raises(RuntimeError):
            cosmo.comoving_distance_z1z2(tensor((1, 2)), tensor((3, 4, 5)))

        assert cosmo.comoving_distance_z1z2(1, 2) == - cosmo.comoving_distance_z1z2(2, 1)
        assert close(
            cosmo.comoving_distance_z1z2(tensor([0, 0, 2, 0.5, 1]), tensor([2, 1, 1, 2.5, 1.1])).to(Mpc).value,
            [3767.90579253, 2386.25591391, -1381.64987862, 2893.11776663, 174.1524683]
        )

    @with_default_double
    @mark.parametrize('Om0, val', (
        # (0, 2997.92458),  # TODO: cannot do Om0=0 with LambdaCDM, need special cosmology
        (1, 1756.1435599923348),
    ))
    def test_distance_in_special_cosmologies(self, Om0, val):
        cosmo = self.flat_cosmo_cls(Om0=Om0)

        assert close(cosmo.comoving_distance(0).to(Mpc).value, 0)
        assert close(cosmo.comoving_distance(1).to(Mpc).value, val)

    @with_default_double
    def test_comoving_transverse_distance_z1z2(self):
        z1, z2 = tensor([0, 0, 2, 0.5, 1]), tensor([2, 1, 1, 2.5, 1.1])

        cosmo = self.flat_cosmo_cls(Om0=0.3)

        with pytest.raises(RuntimeError):
            cosmo.comoving_transverse_distance_z1z2(tensor((1, 2)), tensor((3, 4, 5)))

        assert close(cosmo.comoving_transverse_distance_z1z2(1, 2).to(Mpc).value, 1313.2232194828466)

        assert close(cosmo.comoving_distance_z1z2(z1, z2).to(Mpc).value,
                     cosmo.comoving_transverse_distance_z1z2(z1, z2).to(Mpc).value)

        cosmo = self.flat_cosmo_cls(Om0=1.5)
        assert close(
            cosmo.comoving_transverse_distance_z1z2(z1, z2).to(Mpc).value,
            [2202.72682564, 1559.51679971, -643.21002593, 1408.36365679, 85.09286258]
        )
        assert close(cosmo.comoving_distance_z1z2(z1, z2).to(Mpc).value,
                     cosmo.comoving_transverse_distance_z1z2(z1, z2).to(Mpc).value)

        cosmo = self.cosmo_cls(Om0=0.3, Ode0=0.5)
        assert close(
            cosmo.comoving_transverse_distance_z1z2(z1, z2).to(Mpc).value,
            [3535.931375645655, 2226.430046551708, -1208.6817970036532, 2595.567367601969, 151.36592003406884]
        )

        cosmo = self.cosmo_cls(Om0=1, Ode0=0.2)
        assert close(
            cosmo.comoving_transverse_distance_z1z2(0.1, tensor([0, 0.1, 0.2, 0.5, 1.1, 2])).to(Mpc).value,
            [-281.31602666724865, 0, 248.58093707820436, 843.9331377460543, 1618.6104987686672, 2287.5626543279927]
        )

    def test_angular_diameter_distance_z1z2(self):
        cosmo = self.flat_cosmo_cls(H0=H704, Om0=0.272)

        with pytest.raises(RuntimeError):
            cosmo.angular_diameter_distance_z1z2(tensor((1, 2)), tensor((3, 4, 5)))

        assert close(cosmo.angular_diameter_distance_z1z2(1, 2).to(Mpc).value, 646.22968662822018)
        assert close(
            cosmo.angular_diameter_distance_z1z2(tensor([0, 0, 2, 0.5, 1]), tensor([2, 1, 1, 2.5, 1.1])).to(Mpc).value,
            [1760.0628637762106, 1670.7497657219858, -969.34452994, 1159.0970895962193, 115.72768186186921]
        )
        assert close(
            cosmo.angular_diameter_distance_z1z2(0.1, tensor([0.1, 0.2, 0.5, 1.1, 2])).to(Mpc).value,
            [0, 332.09893173, 986.35635069, 1508.37010062, 1621.07937976]
        )

        # Non-flat (positive Ok0) test
        cosmo = self.cosmo_cls(H0=H704, Om0=0.2, Ode0=0.5)
        assert close(cosmo.angular_diameter_distance_z1z2(1, 2).to(Mpc).value, 620.1175337852428)

        # Non-flat (negative Ok0) test
        cosmo = self.cosmo_cls(Om0=2, Ode0=1)
        assert close(cosmo.angular_diameter_distance_z1z2(1, 2).to(Mpc).value, 228.42914659246014)

    def test_absorption_distance(self):
        cosmo = self.flat_cosmo_cls(H0=H704, Om0=0.272)
        assert close(cosmo.absorption_distance(3), 7.98685853)
        assert close(cosmo.absorption_distance(tensor([1, 3])), [1.72576635, 7.98685853])


class BaseLambdaCDMRTest(BaseLambdaCDMDriverTest):
    flat_cosmo_cls: Type[special.FlatLambdaCDMR]
    cosmo_cls: Type[special.LambdaCDMR]

    @with_default_double
    def test_ogamma(self):
        z = tensor([1, 10, 500, 1000])

        for Neff, Tcmb0, vals in (
            # (3, 0, [1651.9, 858.2, 26.855, 13.642]),  # cannot have Or0=0
            (3, 2.725, [1651.8, 857.9, 26.767, 13.582]),
            (3, 4, [1651.4, 856.6, 26.489, 13.405]),
            # (3.04, 0, [1651.91, 858.205, 26.8586, 13.6469]),  # cannot have Or0=0
            (3.04, 2.725, [1651.76, 857.817, 26.7688, 13.5841]),
            (3.04, 4, [1651.21, 856.411, 26.4845, 13.4028]),
        ):
            assert close(self.flat_cosmo_cls(
                H0=H70, Om0=0.3, Neff=Neff, Tcmb0=Tcmb0*kelvin
            ).angular_diameter_distance(z).to(Mpc).value, vals, rtol=5e-4)

        # from astropy: Just to be really sure, we also do a version where the
        # integral is analytic, which is a Ode = 0 flat universe. In this case
        # Integrate(1/E(x),{x,0,z}) = 2 ( sqrt((1+Or z)/(1+z)) - 1 )/(Or - 1)
        # Recall that c/H0 * Integrate(1/E) is FLRW.comoving_distance.
        hubdis = (299792.458 / 70.0)
        Neff = 3.04
        for Tcmb0 in (2.725, 5):
            Ogamma0h2 = 4 * 5.670373e-8 / 299792458**3 * Tcmb0**4 / 1.87837e-26
            Onu0h2 = Ogamma0h2 * 7/8 * (4 / 11)**(4/3) * Neff
            Or0 = (Ogamma0h2 + Onu0h2) / 0.7**2
            vals = 2 * hubdis * (((1 + Or0*z) / (1+z))**0.5 - 1) / (Or0 - 1)

            cosmo = self.flat_cosmo_cls(H0=H70, Om0=1.)
            cosmo.Neff, cosmo.Tcmb0, cosmo.Ode0 = Neff, Tcmb0*kelvin, 0.
            assert close(cosmo.comoving_distance(z).to(Mpc).value, vals)


# TODO: generate cosmology driver tests automatically
for driver in (analytic, analytic_diff, odeint):
    name = re.sub(r'(?:(?<=^).)|_.', lambda m: m.group(0)[-1].upper(), driver.__name__.rsplit('.', 1)[-1])
    globals()[n] = type(n := f'Test{name}LambdaCDM', (BaseLambdaCDMTest,), dict(
        flat_cosmo_cls=driver.FlatLambdaCDM,
        cosmo_cls=driver.LambdaCDM
    ))
    globals()[n] = type(n := f'Test{name}LambdaCDMR', (BaseLambdaCDMRTest,), dict(
        flat_cosmo_cls=driver.FlatLambdaCDMR,
        cosmo_cls=driver.LambdaCDMR
    ))


# TODO: (age...) test_age
# TODO: (age...) test_age_in_special_cosmologies
# TODO: (neutrinos, weird models...) test_distances
