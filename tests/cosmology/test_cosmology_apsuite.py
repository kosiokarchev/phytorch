# Based on the astropy test suite (v4.2.1)
# (https://github.com/astropy/astropy/blob/v4.2.1/astropy/cosmology/tests/test_cosmology.py)
from io import StringIO
from math import nan
from typing import Type

import numpy as np
import pytest
import torch
from torch import allclose, isclose, tensor

import phytorch.cosmology.drivers.analytic
import phytorch.cosmology.drivers.analytic_diff
import phytorch.cosmology.special
from phytorch.constants import codata2014, G as Newton_G
from phytorch.cosmology.core import FLRW
from phytorch.units.astro import Gpc, Gyr, Mpc
from phytorch.units.si import cm, gram, kelvin, km, s
from phytorch.units.Unit import Unit
from phytorch.utils._typing import _TN
from tests.common import with_default_double


class AbstractLambdaCDMR(phytorch.cosmology.special.LambdaCDMR):
    def lookback_time_dimless(self, z: _TN) -> _TN: ...
    def age_dimless(self, z: _TN) -> _TN: ...
    def absorption_distance_dimless(self, z: _TN) -> _TN: ...


ZERO = torch.zeros(())
ONE = torch.ones(())
SMALL = 1e-16
Z = tensor([0, 0.5, 1, 2])

H100 = 100 * km/s/Mpc
H70 = 70 * km/s/Mpc
H704 = 70.4 * km/s/Mpc


def test_critical_density():
    fac = (Newton_G / codata2014.G).to(Unit())

    # TODO: add flat
    cosmo = AbstractLambdaCDMR()
    cosmo.H0 = H704
    cosmo.Om0 = 0.272
    cosmo.Ode0 = 0.728

    # constants defined only so accurately
    assert ((cosmo.critical_density0 * fac).to(gram / cm**3) - 9.309668456020899e-30) < 1e-9
    assert cosmo.critical_density0 == cosmo.critical_density(0)

    assert allclose((cosmo.critical_density(tensor([1, 5])) * fac).to(gram / cm**3).value,
                    tensor([2.70352772e-29, 5.53739080e-28]))


def test_xtfuncs():
    cosmo = AbstractLambdaCDMR()
    cosmo.H0, cosmo.Om0, cosmo.Ode0, cosmo.Neff, cosmo.Tcmb0 = H70, 0.3, 0.5, 3.04, 2.725 * kelvin

    z = tensor([2, 3.2])
    assert allclose(cosmo.lookback_time_integrand(tensor(3)), tensor(0.052218976654969378))
    assert allclose(cosmo.lookback_time_integrand(z), tensor([0.10333179, 0.04644541]))
    assert allclose(cosmo.abs_distance_integrand(tensor(3)), tensor(3.3420145059180402))
    assert allclose(cosmo.abs_distance_integrand(z), tensor([2.7899584, 3.44104758]),)


def test_zeroing():
    cosmo = AbstractLambdaCDMR()
    cosmo.Om0 = 0.27
    cosmo.Ode0 = 0
    cosmo.Or0 = 0

    assert cosmo.Ode(1.5) == 0
    assert (cosmo.Ode(Z) == ZERO).all()
    assert cosmo.Or(1.5) == 0
    assert (cosmo.Or(Z) == ZERO).all()
    # TODO: add neutrinos
    # assert allclose(cosmo.Onu(1.5), [0, 0, 0, 0])
    # assert allclose(cosmo.Onu(z), [0, 0, 0, 0])
    assert (cosmo.Ob(Z) == ZERO).all()


def test_matter():
    # TODO: add flat
    cosmo = AbstractLambdaCDMR()
    cosmo.Om0 = 0.3
    cosmo.Ode0 = 0.7
    cosmo.Ob0 = 0.045

    assert cosmo.Om(0) == 0.3
    assert cosmo.Ob(0) == 0.045
    assert allclose(cosmo.Om(Z), tensor([0.3, 0.59124088, 0.77419355, 0.92045455]))
    assert allclose(cosmo.Ob(Z), tensor([0.045, 0.08868613, 0.11612903, 0.13806818]))
    assert allclose(cosmo.Odm(Z), tensor([0.255, 0.50255474, 0.65806452, 0.78238636]))
    assert allclose(cosmo.Ob(Z) + cosmo.Odm(Z), cosmo.Om(Z))


def test_ocurv():
    # TODO: add flat
    cosmo = AbstractLambdaCDMR()
    cosmo.Om0 = 0.3
    cosmo.Ode0 = 0.7

    assert cosmo.Ok0 == 0
    assert cosmo.Ok(0) == 0
    assert (cosmo.Ok(Z) == ZERO).all()

    cosmo.Ode0 = 0.5
    assert abs(cosmo.Ok0 - 0.2) < SMALL
    assert abs(cosmo.Ok(0) - 0.2) < SMALL
    assert allclose(cosmo.Ok(Z), tensor([0.2, 0.22929936, 0.21621622, 0.17307692]))

    assert (cosmo.Ok(Z) + cosmo.Om(Z) + cosmo.Ode(Z) == ONE).all()


def test_ode():
    # TODO: add flat
    cosmo = AbstractLambdaCDMR()
    cosmo.Om0 = 0.3
    cosmo.Ode0 = 0.7

    assert cosmo.Ode(0) == cosmo.Ode0
    assert allclose(cosmo.Ode(Z), tensor([0.7, 0.408759, 0.2258065, 0.07954545]))


def test_tcmb():
    # TODO: add flat
    cosmo = AbstractLambdaCDMR()
    cosmo.H0 = H704
    cosmo.Om0 = 0.272
    cosmo.Tcmb0 = 2.5 * kelvin
    cosmo.Ode0 = 1 - cosmo.Om0 - cosmo.Or0

    assert cosmo.Tcmb(2) == 7.5 * kelvin
    assert (cosmo.Tcmb(tensor([0, 1, 2, 3, 9.])).to(kelvin).value == tensor([2.5, 5, 7.5, 10, 25])).all()


def test_efunc_vs_invefunc():
    cosmo = AbstractLambdaCDMR()
    cosmo.Om0 = 0.3
    cosmo.Ode0 = 0.7

    assert cosmo.efunc(0.5) * cosmo.inv_efunc(0.5) == 1
    assert (cosmo.efunc(Z) * cosmo.inv_efunc(Z) == ONE).all()
    # TODO: test this for subclasses?


class BaseDriverTest:
    cosmo_cls: Type[FLRW]


class BaseCDMTest(BaseDriverTest):
    def test_flat_z1(self):
        # TODO: add flat
        cosmo = self.cosmo_cls()
        cosmo.H0 = H70
        cosmo.Om0 = 0.27
        cosmo.Ode0 = 0.73

        # From the astropy test suite:
        # Test values were taken from the following web cosmology
        # calculators on 27th Feb 2012:

        # Wright: http://www.astro.ucla.edu/~wright/CosmoCalc.html
        #         (https://ui.adsabs.harvard.edu/abs/2006PASP..118.1711W)
        # Kempner: http://www.kempner.net/cosmic.php
        # iCosmos: http://www.icosmos.co.uk/index.html
        for func, vals, unit, rtol in (
            (cosmo.comoving_distance, [3364.5, 3364.8, 3364.7988], Mpc, 1e-4),
            (cosmo.angular_diameter_distance, [1682.3, 1682.4, 1682.3994], Mpc, 1e-4),
            (cosmo.luminosity_distance, [6729.2, 6729.6, 6729.5976], Mpc, 1e-4),
            (cosmo.lookback_time, [7.841, 7.84178, 7.843], Gyr, 1e-3),
            (cosmo.lookback_distance, [2404.0, 2404.24, 2404.4], Mpc, 1e-3)
        ):
            assert allclose(func(1).to(unit).value, tensor(vals), rtol=rtol)

    def test_comoving_volume(self):
        z = tensor([0.5, 1, 2, 3, 5, 9])
        for (Om0, Ode0), vals in zip(
            ((0.27, 0.73), (0.27, 0), (2, 0)),
            # Form Ned Wright's calculator: not very *accurate* (sic), so
            # like astropy, test to very low precision
            ((29.123, 159.529, 630.427, 1178.531, 2181.485, 3654.802),
             (20.501, 99.019, 380.278, 747.049, 1558.363, 3123.814),
             (12.619, 44.708, 114.904, 173.709, 258.82, 358.992))
        ):
            c = self.cosmo_cls()
            c.H0, c.Om0, c.Ode0 = H70, Om0, Ode0

            assert allclose(c.comoving_volume(z).to(Gpc**3).value, tensor(vals), rtol=1e-2)

    # TODO: (requires integration) test_differential_comoving_volume

    def test_flat_open_closed_icosmo(self):
        cosmo_flat = """\
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

        cosmo_open = """\
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

        cosmo_closed = """\
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

        for Om0, Ode0, data in ((0.3, 0.7, cosmo_flat), (0.3, 0.1, cosmo_open), (2, 0.1, cosmo_closed)):
            cosmo = self.cosmo_cls()
            cosmo.H0, cosmo.Om0, cosmo.Ode0 = H70, Om0, Ode0
            z, dm, da, dl = (tensor(_, dtype=torch.get_default_dtype())[1:]  # TODO: ER x=y
                             for _ in np.loadtxt(StringIO(data), unpack=True))
            assert allclose(cosmo.comoving_transverse_distance(z).to(Mpc).value, dm)
            assert allclose(cosmo.angular_diameter_distance(z).to(Mpc).value, da)
            assert allclose(cosmo.luminosity_distance(z).to(Mpc).value, dl)

    def test_distmod(self):
        # TODO: add flat
        cosmo = self.cosmo_cls()
        cosmo.H0, cosmo.Om0, cosmo.Ode0 = H704, 0.272, 0.728

        assert cosmo.hubble_distance.to(Mpc) == 4258.415596590909
        assert allclose(cosmo.distmod(tensor([1, 5])), tensor([44.124857, 48.40167258]))

    @with_default_double
    def test_negdistmod(self):
        # TODO: add flat
        cosmo = self.cosmo_cls()
        cosmo.H0, cosmo.Om0, cosmo.Ode0 = H70, 0.2, 1.3
        z = tensor([50, 100])
        assert allclose(cosmo.luminosity_distance(z).to(Mpc).value, tensor([16612.44047622, -46890.79092244]))
        assert allclose(cosmo.distmod(z), tensor([46.102167189, 48.355437790944]))

    def test_comoving_distance_z1z2(self):
        cosmo = self.cosmo_cls()
        cosmo.H0, cosmo.Om0, cosmo.Ode0 = H100, 0.3, 0.8

        with pytest.raises(RuntimeError):
            cosmo.comoving_distance_z1z2(tensor((1, 2)), tensor((3, 4, 5)))

        # TODO: ER x<y
        # assert cosmo.comoving_distance_z1z2(1, 2) == -cosmo.comoving_distance_z1z2(z2, z1)

        # TODO: ER x<y
        assert allclose(
            cosmo.comoving_distance_z1z2(tensor([0, 0, 2, 0.5, 1]), tensor([2, 1, 1, 2.5, 1.1])).to(Mpc).value,
            abs(tensor([3767.90579253, 2386.25591391, -1381.64987862, 2893.11776663, 174.1524683]))
        )

    @with_default_double
    def test_distance_in_special_cosmologies(self):
        # 1. cannot do Om0=0 with LambdaCDM, need special cosmology
        # TODO: add flat
        for Om0, Ode0, val in (
                # (tensor(0), tensor(1), 2997.92458),
                (tensor(1), tensor(0), 1756.1435599923348),):
            cosmo = self.cosmo_cls()
            cosmo.H0, cosmo.Om0, cosmo.Ode0 = H100, Om0, Ode0

            assert allclose(
                cosmo.comoving_distance(tensor([0])).to(Mpc).value,
                tensor(nan),
                equal_nan=True  # TODO: ER x=y
            )
            assert isclose(cosmo.comoving_distance(tensor([1])).to(Mpc).value, tensor(val))

    @with_default_double
    def test_comoving_transverse_distance_z1z2(self):
        z1, z2 = tensor([0, 0, 2, 0.5, 1]), tensor([2, 1, 1, 2.5, 1.1])

        # TODO: add flat
        cosmo = self.cosmo_cls()
        cosmo.H0, cosmo.Om0, cosmo.Ode0 = H100, 0.3, 0.7

        with pytest.raises(RuntimeError):
            cosmo.comoving_transverse_distance_z1z2(tensor((1, 2)), tensor((3, 4, 5)))

        assert isclose(cosmo.comoving_transverse_distance_z1z2(1, 2).to(Mpc).value, tensor(1313.2232194828466))

        assert allclose(cosmo.comoving_distance_z1z2(z1, z2).to(Mpc).value,
                        cosmo.comoving_transverse_distance_z1z2(z1, z2).to(Mpc).value)

        # TODO: add flat
        cosmo = self.cosmo_cls()
        cosmo.H0, cosmo.Om0, cosmo.Ode0 = H100, 1.5, -0.5
        # TODO: ER x<y
        assert allclose(
            cosmo.comoving_transverse_distance_z1z2(z1, z2).to(Mpc).value,
            abs(tensor([2202.72682564, 1559.51679971, -643.21002593, 1408.36365679, 85.09286258]))
        )
        assert allclose(cosmo.comoving_distance_z1z2(z1, z2).to(Mpc).value,
                        cosmo.comoving_transverse_distance_z1z2(z1, z2).to(Mpc).value)

        cosmo = self.cosmo_cls()
        cosmo.H0, cosmo.Om0, cosmo.Ode0 = H100, 0.3, 0.5
        # TODO: ER x<y
        assert allclose(
            cosmo.comoving_transverse_distance_z1z2(z1, z2).to(Mpc).value,
            abs(tensor([3535.931375645655, 2226.430046551708, -1208.6817970036532, 2595.567367601969, 151.36592003406884]))
        )

        cosmo = self.cosmo_cls()
        cosmo.H0, cosmo.Om0, cosmo.Ode0 = H100, 1, 0.2
        # TODO: ER x<y
        assert allclose(
            cosmo.comoving_transverse_distance_z1z2(0.1, tensor([0, 0.1, 0.2, 0.5, 1.1, 2])).to(Mpc).value,
            abs(tensor([-281.31602666724865, nan, 248.58093707820436, 843.9331377460543, 1618.6104987686672, 2287.5626543279927])),
            equal_nan=True  # TODO: ER x=y
        )

    def test_angular_diameter_distance_z1z2(self):
        # TODO: add flat
        cosmo = self.cosmo_cls()
        cosmo.H0, cosmo.Om0, cosmo.Ode0 = H704, 0.272, 0.728

        with pytest.raises(RuntimeError):
            cosmo.angular_diameter_distance_z1z2(tensor((1, 2)), tensor((3, 4, 5)))

        assert isclose(cosmo.angular_diameter_distance_z1z2(1, 2).to(Mpc).value,
                       tensor(646.22968662822018))

        # TODO: ER x-y symmetry
        assert allclose(
            cosmo.angular_diameter_distance_z1z2(tensor([0, 0, 2, 0.5, 1]), tensor([2, 1, 1, 2.5, 1.1])).to(Mpc).value,
            abs(tensor([1760.0628637762106, 1670.7497657219858, -969.34452994, 1159.0970895962193, 115.72768186186921]))
        )

        assert allclose(
            cosmo.angular_diameter_distance_z1z2(0.1, tensor([0.1, 0.2, 0.5, 1.1, 2])).to(Mpc).value,
            tensor([nan, 332.09893173, 986.35635069, 1508.37010062, 1621.07937976]),
            equal_nan=True  # TODO: ER x=y
        )

        # Non-flat (positive Ok0) test
        cosmo = self.cosmo_cls()
        cosmo.H0, cosmo.Om0, cosmo.Ode0 = H704, 0.2, 0.5
        assert isclose(cosmo.angular_diameter_distance_z1z2(1, 2).to(Mpc).value, tensor(620.1175337852428))

        # Non-flat (negative Ok0) test
        cosmo = self.cosmo_cls()
        cosmo.H0, cosmo.Om0, cosmo.Ode0 = H100, 2, 1
        assert isclose(cosmo.angular_diameter_distance_z1z2(1, 2).to(Mpc).value, tensor(228.42914659246014))

    # TODO: absorption_distance
    # def test_absorption_distance(self):
    #     # TODO: add flat
    #     cosmo = self.cosmo_cls()
    #     cosmo.H0, cosmo.Om0, cosmo.Ode0 = H704, 0.272, 0.728
    #     assert isclose(cosmo.absorption_distance(3), tensor(7.98685853))
    #     assert allclose(cosmo.absorption_distance(tensor([1, 3])), tensor([1.72576635, 7.98685853]))


# this will fail for all tests that have Or0=0
class BaseCDMRTest(BaseDriverTest):
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
            # TODO: add flat
            cosmo = self.cosmo_cls()
            cosmo.H0, cosmo.Om0, cosmo.Neff, cosmo.Tcmb0 = H70, 0.3, Neff, Tcmb0*kelvin
            cosmo.Ode0 = 0
            cosmo.Ode0 = cosmo.Ok0

            assert allclose(cosmo.angular_diameter_distance(z).to(Mpc).value, tensor(vals), rtol=5e-4)

        # from astropy: Just to be really sure, we also do a version where the
        # integral is analytic, which is a Ode = 0 flat universe. In this case
        # Integrate(1/E(x),{x,0,z}) = 2 ( sqrt((1+Or z)/(1+z)) - 1 )/(Or - 1)
        # Recall that c/H0 * Integrate(1/E) is FLRW.comoving_distance.
        hubdis = (299792.458 / 70.0)
        Neff = 3.04
        for Tcmb0 in (2.725, 5):
            Ogamma0h2 = 4 * 5.670373e-8 / 299792458.0**3 * Tcmb0**4 / 1.87837e-26
            Onu0h2 = Ogamma0h2 * 7.0 / 8.0 * (4.0 / 11.0)**(4.0 / 3.0) * Neff
            Or0 = (Ogamma0h2 + Onu0h2) / 0.7**2
            Om0 = 1.0 - Or0
            vals = 2.0 * hubdis * (((1.0 + Or0 * z) / (1.0 + z))**0.5 - 1.0) / (Or0 - 1.0)

            # TODO: add flat
            cosmo = self.cosmo_cls()
            cosmo.H0, cosmo.Om0, cosmo.Ode0, cosmo.Neff, cosmo.Tcmb0 = H70, Om0, 0, Neff, Tcmb0 * kelvin

            assert allclose(cosmo.comoving_distance(z).to(Mpc).value, tensor(vals))


class TestAnalyticCDM(BaseCDMTest):
    cosmo_cls = phytorch.cosmology.drivers.analytic.LambdaCDM


class TestAnalyticCDMR(BaseCDMRTest):
    cosmo_cls = phytorch.cosmology.drivers.analytic.LambdaCDMR


class TestAnalyticDiffCDM(BaseCDMTest):
    cosmo_cls = phytorch.cosmology.drivers.analytic_diff.LambdaCDM


class TestAnalyticDiffCDMR(BaseCDMRTest):
    cosmo_cls = phytorch.cosmology.drivers.analytic_diff.LambdaCDMR


# TODO: (age...) test_age
# TODO: (age...) test_age_in_special_cosmologies
# TODO: (neutrinos, weird models...) test_distances
