// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// reg_F
Eigen::MatrixXd reg_F(const Eigen::Map<Eigen::MatrixXd>& R, const Eigen::Map<Eigen::MatrixXd>& Q, Eigen::Map<Eigen::MatrixXd>& Q_r, const DataFrame& Y, const IntegerVector& IND);
RcppExport SEXP _CppFunc_reg_F(SEXP RSEXP, SEXP QSEXP, SEXP Q_rSEXP, SEXP YSEXP, SEXP INDSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type R(RSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type Q(QSEXP);
    Rcpp::traits::input_parameter< Eigen::Map<Eigen::MatrixXd>& >::type Q_r(Q_rSEXP);
    Rcpp::traits::input_parameter< const DataFrame& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const IntegerVector& >::type IND(INDSEXP);
    rcpp_result_gen = Rcpp::wrap(reg_F(R, Q, Q_r, Y, IND));
    return rcpp_result_gen;
END_RCPP
}
// reg_F2
Eigen::MatrixXd reg_F2(const Eigen::Map<Eigen::MatrixXd>& R, const Eigen::Map<Eigen::MatrixXd>& Q, Eigen::Map<Eigen::MatrixXd>& Q_r, const DataFrame& Y);
RcppExport SEXP _CppFunc_reg_F2(SEXP RSEXP, SEXP QSEXP, SEXP Q_rSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type R(RSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type Q(QSEXP);
    Rcpp::traits::input_parameter< Eigen::Map<Eigen::MatrixXd>& >::type Q_r(Q_rSEXP);
    Rcpp::traits::input_parameter< const DataFrame& >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(reg_F2(R, Q, Q_r, Y));
    return rcpp_result_gen;
END_RCPP
}
// reg_F3
Eigen::MatrixXd reg_F3(const Eigen::Map<Eigen::MatrixXd>& R, const Eigen::Map<Eigen::MatrixXd>& Q, Eigen::Map<Eigen::MatrixXd>& Q_r, const DataFrame& Y, const int& K);
RcppExport SEXP _CppFunc_reg_F3(SEXP RSEXP, SEXP QSEXP, SEXP Q_rSEXP, SEXP YSEXP, SEXP KSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type R(RSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type Q(QSEXP);
    Rcpp::traits::input_parameter< Eigen::Map<Eigen::MatrixXd>& >::type Q_r(Q_rSEXP);
    Rcpp::traits::input_parameter< const DataFrame& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const int& >::type K(KSEXP);
    rcpp_result_gen = Rcpp::wrap(reg_F3(R, Q, Q_r, Y, K));
    return rcpp_result_gen;
END_RCPP
}
// reg_F4
Eigen::MatrixXd reg_F4(const Eigen::Map<Eigen::MatrixXd>& R, const Eigen::Map<Eigen::MatrixXd>& Q, Eigen::Map<Eigen::MatrixXd>& Q_r, const DataFrame& Y, const IntegerVector& IND, const int& K);
RcppExport SEXP _CppFunc_reg_F4(SEXP RSEXP, SEXP QSEXP, SEXP Q_rSEXP, SEXP YSEXP, SEXP INDSEXP, SEXP KSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type R(RSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type Q(QSEXP);
    Rcpp::traits::input_parameter< Eigen::Map<Eigen::MatrixXd>& >::type Q_r(Q_rSEXP);
    Rcpp::traits::input_parameter< const DataFrame& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const IntegerVector& >::type IND(INDSEXP);
    Rcpp::traits::input_parameter< const int& >::type K(KSEXP);
    rcpp_result_gen = Rcpp::wrap(reg_F4(R, Q, Q_r, Y, IND, K));
    return rcpp_result_gen;
END_RCPP
}
// reg_cov
Eigen::MatrixXd reg_cov(const Eigen::Map<Eigen::MatrixXd>& R, const Eigen::Map<Eigen::MatrixXd>& Q, const DataFrame& Y, const IntegerVector& IND, const int& K);
RcppExport SEXP _CppFunc_reg_cov(SEXP RSEXP, SEXP QSEXP, SEXP YSEXP, SEXP INDSEXP, SEXP KSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type R(RSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type Q(QSEXP);
    Rcpp::traits::input_parameter< const DataFrame& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const IntegerVector& >::type IND(INDSEXP);
    Rcpp::traits::input_parameter< const int& >::type K(KSEXP);
    rcpp_result_gen = Rcpp::wrap(reg_cov(R, Q, Y, IND, K));
    return rcpp_result_gen;
END_RCPP
}
// reg_F_ROI
Eigen::MatrixXd reg_F_ROI(const Eigen::Map<Eigen::MatrixXd>& R, const Eigen::Map<Eigen::MatrixXd>& Q, const Eigen::Map<Eigen::MatrixXd>& Q_r, const Eigen::Map<Eigen::MatrixXd> Y);
RcppExport SEXP _CppFunc_reg_F_ROI(SEXP RSEXP, SEXP QSEXP, SEXP Q_rSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type R(RSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type Q(QSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type Q_r(Q_rSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd> >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(reg_F_ROI(R, Q, Q_r, Y));
    return rcpp_result_gen;
END_RCPP
}
// res_vox
Eigen::MatrixXd res_vox(const Eigen::Map<Eigen::MatrixXd>& Q, const DataFrame& Y, const IntegerVector& IND);
RcppExport SEXP _CppFunc_res_vox(SEXP QSEXP, SEXP YSEXP, SEXP INDSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type Q(QSEXP);
    Rcpp::traits::input_parameter< const DataFrame& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const IntegerVector& >::type IND(INDSEXP);
    rcpp_result_gen = Rcpp::wrap(res_vox(Q, Y, IND));
    return rcpp_result_gen;
END_RCPP
}
// res_ROI
Eigen::MatrixXd res_ROI(const Eigen::Map<Eigen::MatrixXd>& Q, const DataFrame& Y);
RcppExport SEXP _CppFunc_res_ROI(SEXP QSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type Q(QSEXP);
    Rcpp::traits::input_parameter< const DataFrame& >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(res_ROI(Q, Y));
    return rcpp_result_gen;
END_RCPP
}
// perm_mri
Eigen::MatrixXd perm_mri(const Eigen::Map<Eigen::MatrixXd>& Q, const Eigen::Map<Eigen::MatrixXd>& Q_r, const Eigen::Map<Eigen::MatrixXd> Y, const int DF);
RcppExport SEXP _CppFunc_perm_mri(SEXP QSEXP, SEXP Q_rSEXP, SEXP YSEXP, SEXP DFSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type Q(QSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type Q_r(Q_rSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd> >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const int >::type DF(DFSEXP);
    rcpp_result_gen = Rcpp::wrap(perm_mri(Q, Q_r, Y, DF));
    return rcpp_result_gen;
END_RCPP
}
// IV_F
Eigen::MatrixXd IV_F(const Eigen::Map<Eigen::MatrixXd>& X, const Eigen::Map<Eigen::MatrixXd>& R, const Eigen::Map<Eigen::MatrixXd>& Q, Eigen::Map<Eigen::MatrixXd>& Q_r, const DataFrame& Y, const IntegerVector& IND, const int& K);
RcppExport SEXP _CppFunc_IV_F(SEXP XSEXP, SEXP RSEXP, SEXP QSEXP, SEXP Q_rSEXP, SEXP YSEXP, SEXP INDSEXP, SEXP KSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type R(RSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type Q(QSEXP);
    Rcpp::traits::input_parameter< Eigen::Map<Eigen::MatrixXd>& >::type Q_r(Q_rSEXP);
    Rcpp::traits::input_parameter< const DataFrame& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const IntegerVector& >::type IND(INDSEXP);
    Rcpp::traits::input_parameter< const int& >::type K(KSEXP);
    rcpp_result_gen = Rcpp::wrap(IV_F(X, R, Q, Q_r, Y, IND, K));
    return rcpp_result_gen;
END_RCPP
}
// lm_cpp
List lm_cpp(const Eigen::Map<Eigen::MatrixXd>& X, const Eigen::Map<Eigen::MatrixXd>& Y);
RcppExport SEXP _CppFunc_lm_cpp(SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(lm_cpp(X, Y));
    return rcpp_result_gen;
END_RCPP
}
// lm_cpp_het
List lm_cpp_het(const Eigen::Map<Eigen::MatrixXd>& X, const Eigen::Map<Eigen::MatrixXd>& Y);
RcppExport SEXP _CppFunc_lm_cpp_het(SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(lm_cpp_het(X, Y));
    return rcpp_result_gen;
END_RCPP
}
// lm_cpp_cluster
List lm_cpp_cluster(const Eigen::Map<Eigen::MatrixXd>& X, const Eigen::Map<Eigen::MatrixXd>& Y, const NumericVector& C);
RcppExport SEXP _CppFunc_lm_cpp_cluster(SEXP XSEXP, SEXP YSEXP, SEXP CSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type C(CSEXP);
    rcpp_result_gen = Rcpp::wrap(lm_cpp_cluster(X, Y, C));
    return rcpp_result_gen;
END_RCPP
}
// iv_cpp
List iv_cpp(const Eigen::Map<Eigen::MatrixXd>& X, const Eigen::Map<Eigen::MatrixXd>& Z, const Eigen::Map<Eigen::MatrixXd>& Y);
RcppExport SEXP _CppFunc_iv_cpp(SEXP XSEXP, SEXP ZSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(iv_cpp(X, Z, Y));
    return rcpp_result_gen;
END_RCPP
}
// iv_cpp_het
List iv_cpp_het(const Eigen::Map<Eigen::MatrixXd>& X, const Eigen::Map<Eigen::MatrixXd>& Z, const Eigen::Map<Eigen::MatrixXd>& Y);
RcppExport SEXP _CppFunc_iv_cpp_het(SEXP XSEXP, SEXP ZSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(iv_cpp_het(X, Z, Y));
    return rcpp_result_gen;
END_RCPP
}
// iv_cpp_cluster
List iv_cpp_cluster(const Eigen::Map<Eigen::MatrixXd>& X, const Eigen::Map<Eigen::MatrixXd>& Z, const Eigen::Map<Eigen::MatrixXd>& Y, const NumericVector& C);
RcppExport SEXP _CppFunc_iv_cpp_cluster(SEXP XSEXP, SEXP ZSEXP, SEXP YSEXP, SEXP CSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type C(CSEXP);
    rcpp_result_gen = Rcpp::wrap(iv_cpp_cluster(X, Z, Y, C));
    return rcpp_result_gen;
END_RCPP
}
// RIDGE_K
Eigen::MatrixXd RIDGE_K(const Eigen::Map<Eigen::MatrixXd>& X, const Eigen::Map<Eigen::MatrixXd>& Y, const int& K);
RcppExport SEXP _CppFunc_RIDGE_K(SEXP XSEXP, SEXP YSEXP, SEXP KSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const int& >::type K(KSEXP);
    rcpp_result_gen = Rcpp::wrap(RIDGE_K(X, Y, K));
    return rcpp_result_gen;
END_RCPP
}
// RIDGE_IV_K
List RIDGE_IV_K(const Eigen::Map<Eigen::MatrixXd>& X_input, const Eigen::Map<Eigen::MatrixXd>& Z_input, const Eigen::Map<Eigen::MatrixXd>& Y_input, const int& K);
RcppExport SEXP _CppFunc_RIDGE_IV_K(SEXP X_inputSEXP, SEXP Z_inputSEXP, SEXP Y_inputSEXP, SEXP KSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X_input(X_inputSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type Z_input(Z_inputSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type Y_input(Y_inputSEXP);
    Rcpp::traits::input_parameter< const int& >::type K(KSEXP);
    rcpp_result_gen = Rcpp::wrap(RIDGE_IV_K(X_input, Z_input, Y_input, K));
    return rcpp_result_gen;
END_RCPP
}
// Stdz
Eigen::MatrixXd Stdz(const Eigen::Map<Eigen::MatrixXd>& mat);
RcppExport SEXP _CppFunc_Stdz(SEXP matSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type mat(matSEXP);
    rcpp_result_gen = Rcpp::wrap(Stdz(mat));
    return rcpp_result_gen;
END_RCPP
}
// make_res
Eigen::MatrixXd make_res(const Eigen::Map<Eigen::MatrixXd>& X, const Eigen::Map<Eigen::MatrixXd>& Y);
RcppExport SEXP _CppFunc_make_res(SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(make_res(X, Y));
    return rcpp_result_gen;
END_RCPP
}
// get_XXt
Eigen::MatrixXd get_XXt(const Eigen::Map<Eigen::MatrixXd>& X);
RcppExport SEXP _CppFunc_get_XXt(SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(get_XXt(X));
    return rcpp_result_gen;
END_RCPP
}
// get_XtX
Eigen::MatrixXd get_XtX(const Eigen::Map<Eigen::MatrixXd>& X);
RcppExport SEXP _CppFunc_get_XtX(SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd>& >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(get_XtX(X));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_CppFunc_reg_F", (DL_FUNC) &_CppFunc_reg_F, 5},
    {"_CppFunc_reg_F2", (DL_FUNC) &_CppFunc_reg_F2, 4},
    {"_CppFunc_reg_F3", (DL_FUNC) &_CppFunc_reg_F3, 5},
    {"_CppFunc_reg_F4", (DL_FUNC) &_CppFunc_reg_F4, 6},
    {"_CppFunc_reg_cov", (DL_FUNC) &_CppFunc_reg_cov, 5},
    {"_CppFunc_reg_F_ROI", (DL_FUNC) &_CppFunc_reg_F_ROI, 4},
    {"_CppFunc_res_vox", (DL_FUNC) &_CppFunc_res_vox, 3},
    {"_CppFunc_res_ROI", (DL_FUNC) &_CppFunc_res_ROI, 2},
    {"_CppFunc_perm_mri", (DL_FUNC) &_CppFunc_perm_mri, 4},
    {"_CppFunc_IV_F", (DL_FUNC) &_CppFunc_IV_F, 7},
    {"_CppFunc_lm_cpp", (DL_FUNC) &_CppFunc_lm_cpp, 2},
    {"_CppFunc_lm_cpp_het", (DL_FUNC) &_CppFunc_lm_cpp_het, 2},
    {"_CppFunc_lm_cpp_cluster", (DL_FUNC) &_CppFunc_lm_cpp_cluster, 3},
    {"_CppFunc_iv_cpp", (DL_FUNC) &_CppFunc_iv_cpp, 3},
    {"_CppFunc_iv_cpp_het", (DL_FUNC) &_CppFunc_iv_cpp_het, 3},
    {"_CppFunc_iv_cpp_cluster", (DL_FUNC) &_CppFunc_iv_cpp_cluster, 4},
    {"_CppFunc_RIDGE_K", (DL_FUNC) &_CppFunc_RIDGE_K, 3},
    {"_CppFunc_RIDGE_IV_K", (DL_FUNC) &_CppFunc_RIDGE_IV_K, 4},
    {"_CppFunc_Stdz", (DL_FUNC) &_CppFunc_Stdz, 1},
    {"_CppFunc_make_res", (DL_FUNC) &_CppFunc_make_res, 2},
    {"_CppFunc_get_XXt", (DL_FUNC) &_CppFunc_get_XXt, 1},
    {"_CppFunc_get_XtX", (DL_FUNC) &_CppFunc_get_XtX, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_CppFunc(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
