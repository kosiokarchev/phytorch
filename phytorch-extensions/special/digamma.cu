#include "special.cuh"
#include "gammahead.h"

DEFINE_COMPLEX_FUNCTION(digamma, (z)) {
    // algorithm from https://www.mathworks.com/matlabcentral/fileexchange/978-special-functions-math-library
    if (is_int(z) and is_real_nonpositive(z)) return cnan<T>();

    if (z.real() < 0.5)
        return digamma<scalar_t>(1-z) - ltrl(M_PI) / tan(ltrl(M_PI) * z);

    // TODO: macro-ify
    static scalar_t c[] = {
        0.99999999999999709182,
        57.156235665862923517,
        -59.597960355475491248,
        14.136097974741747174,
        -0.49191381609762019978,
        .33994649984811888699e-4,
        .46523628927048575665e-4,
        -.98374475304879564677e-4,
        .15808870322491248884e-3,
        -.21026444172410488319e-3,
        .21743961811521264320e-3,
        -.16431810653676389022e-3,
        .84418223983852743293e-4,
        -.26190838401581408670e-4,
        .36899182659531622704e-5
    };

    T d = 0, n = 0;
#pragma unroll
    for (auto k=14; k>0; --k) {
        auto dz = ltrl(1) / (z+k-1);
        auto dd = ltrl(c[k]) * dz;
        d += dd;
        n -= dd * dz;
    }
    d += c[0];

    auto zp = z + ltrl(g - h);
    return log(zp) + (n / d - ltrl(g) / zp);
}
