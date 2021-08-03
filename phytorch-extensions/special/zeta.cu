// Algorithms from https://www.mathworks.com/matlabcentral/fileexchange/978-special-functions-math-library
#include "special.cuh"


#define DETA_COEFFS ( \
    .99999999999999999997, -.99999999999999999821, .99999999999999994183, -.99999999999999875788, .99999999999998040668,   \
    -.99999999999975652196, .99999999999751767484, -.99999999997864739190, .99999999984183784058, -.99999999897537734890,  \
    .99999999412319859549, -.99999996986230482845, .99999986068828287678, -.99999941559419338151, .99999776238757525623,   \
    -.99999214148507363026, .99997457616475604912, -.99992394671207596228, .99978893483826239739, -.99945495809777621055,  \
    .99868681159465798081, -.99704078337369034566, .99374872693175507536, -.98759401271422391785, .97682326283354439220,   \
    -.95915923302922997013, .93198380256105393618, -.89273040299591077603, .83945793215750220154, -.77148960729470505477,  \
    .68992761745934847866, -.59784149990330073143, .50000000000000000000, -.40215850009669926857, .31007238254065152134,   \
    -.22851039270529494523, .16054206784249779846, -.10726959700408922397, .68016197438946063823e-1, -.40840766970770029873e-1,              \
    .23176737166455607805e-1, -.12405987285776082154e-1, .62512730682449246388e-2, -.29592166263096543401e-2, .13131884053420191908e-2,      \
    -.54504190222378945440e-3, .21106516173760261250e-3, -.76053287924037718971e-4, .25423835243950883896e-4, -.78585149263697370338e-5,     \
    .22376124247437700378e-5, -.58440580661848562719e-6, .13931171712321674741e-6, -.30137695171547022183e-7, .58768014045093054654e-8,      \
    -.10246226511017621219e-8, .15816215942184366772e-9, -.21352608103961806529e-10, .24823251635643084345e-11, -.24347803504257137241e-12,  \
    .19593322190397666205e-13, -.12421162189080181548e-14, .58167446553847312884e-16, -.17889335846010823161e-17, .27105054312137610850e-19)
#define LIST_TERM(r, data, i, elem) BOOST_PP_IF(i, +, ) ltrl(elem) * pow(ltrl(i+1), -z)

DEFINE_COMPLEX_FUNCTION(deta1, (z)) {
    if (z.real() < 0.5)
        return 2 * (1 - pow(ltrl(2), (z-1))) / (1 - pow(ltrl(2), z)) * pow(ltrl(M_PI), z-1) * z * sin(ltrl(M_PI) * z/2) * gamma<scalar_t>(-z) * deta1<scalar_t>(1-z);
    return BOOST_PP_SEQ_FOR_EACH_I(LIST_TERM, _, BOOST_PP_TUPLE_TO_SEQ(DETA_COEFFS));
}

DEFINE_COMPLEX_FUNCTION(zeta, (z)) {
    if (z == ltrl(1)) return numeric_limits<scalar_t>::infinity();
    return pow(ltrl(2), z) / (pow(ltrl(2), z) - 2) * deta1<scalar_t>(z);
}