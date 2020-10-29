#include <Rcpp.h>
#include <RcppEigen.h>
#include <Rcpp/Benchmark/Timer.h>
#include "utils.hpp"

//[[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;

// [[Rcpp::export]]
List lm_cpp(const Eigen::Map<Eigen::MatrixXd> &X, const Eigen::Map<Eigen::MatrixXd> &Y)
{

  int p = X.cols();
  Eigen::MatrixXd XtY = X.transpose() * Y;
  Eigen::LLT<Eigen::MatrixXd> llt(Eigen::MatrixXd(p, p).setZero().selfadjointView<Eigen::Upper>().rankUpdate(X.adjoint()));
  Eigen::MatrixXd B = llt.solve(XtY);

  // double sigma = (Y - X * B).squaredNorm() / (Y.rows() - p);
  // Eigen::VectorXd SE = sqrt(sigma) * (llt.solve(Eigen::MatrixXd::Identity(p, p))).diagonal().cwiseSqrt();
  Eigen::MatrixXd sigma = ((Y - X * B).colwise().squaredNorm() / (Y.rows() - p)).cwiseSqrt();
  Eigen::MatrixXd SE = (llt.solve(Eigen::MatrixXd::Identity(p, p))).diagonal().cwiseSqrt() * sigma;

  return List::create(Named("BETA") = B,
                      Named("SE") = SE);
  // Named("R2") = 1 - sigma * (Y.rows() - p) / (Y.array() - Y.array().mean()).matrix().squaredNorm() );
}

// [[Rcpp::export]]
List lm_cpp_het(const Eigen::Map<Eigen::MatrixXd> &X, const Eigen::Map<Eigen::MatrixXd> &Y)
{

  int p = X.cols();
  Eigen::MatrixXd XtY = X.transpose() * Y;
  Eigen::LLT<Eigen::MatrixXd> llt_XtX(Eigen::MatrixXd(p, p).setZero().selfadjointView<Eigen::Upper>().rankUpdate(X.adjoint()));
  Eigen::MatrixXd B = llt_XtX.solve(XtY);

  Eigen::MatrixXd INV = llt_XtX.solve(Eigen::MatrixXd::Identity(p, p));
  Eigen::MatrixXd U = (Y - X * B);
  Eigen::MatrixXd SE(X.cols(), Y.cols());
  for (int i = 0; i < SE.cols(); ++i)
  {
    SE.col(i) = Eigen::MatrixXd(p, p).setZero().selfadjointView<Eigen::Upper>().rankUpdate(INV.selfadjointView<Eigen::Upper>() * (X.transpose() * U.col(i).cwiseAbs().asDiagonal())).diagonal().cwiseSqrt();
  }

  return List::create(Named("BETA") = B,
                      Named("SE") = SE);
  // Named("R2") = 1 - sigma * (Y.rows() - p) / (Y.array() - Y.array().mean()).matrix().squaredNorm() );
}

// [[Rcpp::export]]
List lm_cpp_cluster(const Eigen::Map<Eigen::MatrixXd> &X, const Eigen::Map<Eigen::MatrixXd> &Y, const NumericVector &C)
{

  int p = X.cols();
  Eigen::MatrixXd XtY = X.transpose() * Y;
  Eigen::LLT<Eigen::MatrixXd> llt_XtX(Eigen::MatrixXd(p, p).setZero().selfadjointView<Eigen::Upper>().rankUpdate(X.adjoint()));
  Eigen::MatrixXd B = llt_XtX.solve(XtY);

  NumericVector C_u = unique(C);

  Eigen::MatrixXd INV = llt_XtX.solve(Eigen::MatrixXd::Identity(p, p));
  Eigen::MatrixXd U = (Y - X * B);
  Eigen::MatrixXd SE(X.cols(), Y.cols());
  Eigen::MatrixXd Sigma_temp;
  Eigen::MatrixXd VCOV_temp;

  for (int i = 0; i < SE.cols(); ++i)
  {
    NumericVector U_i = wrap(U.col(i));
    Eigen::MatrixXd XtE(Eigen::MatrixXd(p, p).setZero());

    for (int c = 0; c < C_u.size(); ++c)
    {
      // make mask
      NumericVector c_vec(C.size());
      c_vec.fill(C_u[c]);
      LogicalVector mask = (C != c_vec);
        // try change this, using .block
      XtE.selfadjointView<Eigen::Upper>().rankUpdate(submat(X, mask).transpose() * as<Eigen::Map<Eigen::VectorXd>>(U_i[mask]));
    }

    SE.col(i) = (INV.selfadjointView<Eigen::Upper>() * (XtE.selfadjointView<Eigen::Upper>() * INV)).diagonal().cwiseSqrt();
  }

  return List::create(Named("BETA") = B,
                      Named("SE") = SE);
  // Named("R2") = 1 - sigma * (Y.rows() - p) / (Y.array() - Y.array().mean()).matrix().squaredNorm() );
}
// Didn't work! gives same se as het
  // for (int i = 0; i < SE.cols(); ++i)
  // {
  //   NumericVector U_i = wrap(U.col(i));
  //   Eigen::MatrixXd Sigma_i = Eigen::MatrixXd(U_i.size(), U_i.size()).setZero().selfadjointView<Eigen::Upper>();

  //     for (int c = 0; c < C_u.size(); ++c)
  //     {
  //       NumericVector U_i_c = clone(U_i);

  //       //subset U_i for cluster c
  //       NumericVector c_vec(C.size());
  //       c_vec.fill(C_u[c]);
  //       LogicalVector mask = (C != c_vec);
  //       U_i_c[mask] = 0;

  //       Sigma_i.selfadjointView<Eigen::Upper>().rankUpdate(as<Eigen::Map<Eigen::VectorXd>>(U_i_c));
  //     }

  //     SE.col(i) = Eigen::MatrixXd(p, p).setZero().selfadjointView<Eigen::Upper>().rankUpdate(INV.selfadjointView<Eigen::Upper>() * (X.transpose() * Sigma_i.llt().matrixL())).diagonal().cwiseSqrt();    
  // }

// This approach too slow
// for (int i = 0; i < SE.cols(); ++i)
// {
//   NumericVector U_i = wrap(U.col(i));
//   Eigen::MatrixXd XtE(Eigen::MatrixXd(p, p).setZero());

//   for (int c = 0; c < C_u.size(); ++c)
//   {
//     // make mask
//     NumericVector c_vec(C.size());
//     c_vec.fill(C_u[c]);
//     LogicalVector mask = (C != c_vec);

//     XtE.selfadjointView<Eigen::Upper>().rankUpdate(submat(X, mask).transpose() * as<Eigen::Map<Eigen::VectorXd>>(U_i[mask]));
//   }

//   SE.col(i) = (INV.selfadjointView<Eigen::Upper>() * (XtE.selfadjointView<Eigen::Upper>() * INV)).diagonal().cwiseSqrt();
// }

// [[Rcpp::export]]
List iv_cpp(const Eigen::Map<Eigen::MatrixXd> &X, const Eigen::Map<Eigen::MatrixXd> &Z, const Eigen::Map<Eigen::MatrixXd> &Y)
{

  int p = X.cols();

  Eigen::LLT<Eigen::MatrixXd> llt_ZtZ(Eigen::MatrixXd(p, p).setZero().selfadjointView<Eigen::Upper>().rankUpdate(Z.adjoint()));
  // Eigen::MatrixXd PZ = Z * llt.solve(Eigen::MatrixXd::Identity(p, p)).selfadjointView<Eigen::Upper>() * Z.transpose();
  Eigen::MatrixXd Uinv = llt_ZtZ.matrixU().solve(Eigen::MatrixXd::Identity(p, p));
  Eigen::MatrixXd Z_Uinv = Z * Uinv.triangularView<Eigen::Upper>();
  Eigen::MatrixXd Xt_Z_Uinv = X.transpose() * Z_Uinv;
  Eigen::MatrixXd Xt_PZ_X = Eigen::MatrixXd(p, p).setZero().selfadjointView<Eigen::Upper>().rankUpdate(Xt_Z_Uinv);

  Eigen::LLT<Eigen::MatrixXd> llt_denom((Xt_PZ_X).selfadjointView<Eigen::Upper>());
  Eigen::MatrixXd B = llt_denom.solve(Xt_Z_Uinv * (Z_Uinv.transpose() * Y));
  Eigen::MatrixXd sigma = ((Y - X * B).colwise().squaredNorm() / (Y.rows() - p)).cwiseSqrt();
  Eigen::MatrixXd SE = llt_denom.solve(Eigen::MatrixXd::Identity(p, p)).diagonal().cwiseSqrt() * sigma;

  return List::create(Named("BETA") = B,
                      Named("SE") = SE);
}

// [[Rcpp::export]]
List iv_cpp_het(const Eigen::Map<Eigen::MatrixXd> &X, const Eigen::Map<Eigen::MatrixXd> &Z, const Eigen::Map<Eigen::MatrixXd> &Y)
{

  int p = X.cols();

  Eigen::LLT<Eigen::MatrixXd> llt_ZtZ(Eigen::MatrixXd(p, p).setZero().selfadjointView<Eigen::Upper>().rankUpdate(Z.adjoint()));
  // Eigen::MatrixXd PZ = Z * llt.solve(Eigen::MatrixXd::Identity(p, p)).selfadjointView<Eigen::Upper>() * Z.transpose();
  Eigen::MatrixXd Uinv = llt_ZtZ.matrixU().solve(Eigen::MatrixXd::Identity(p, p));
  Eigen::MatrixXd Z_Uinv = Z * Uinv.triangularView<Eigen::Upper>();
  Eigen::MatrixXd Xt_Z_Uinv = X.transpose() * Z_Uinv;
  Eigen::MatrixXd Xt_PZ_X = Eigen::MatrixXd(p, p).setZero().selfadjointView<Eigen::Upper>().rankUpdate(Xt_Z_Uinv);

  Eigen::LLT<Eigen::MatrixXd> llt_denom((Xt_PZ_X).selfadjointView<Eigen::Upper>());
  Eigen::MatrixXd B = llt_denom.solve(Xt_Z_Uinv * (Z_Uinv.transpose() * Y));
  Eigen::MatrixXd U = (Y - X * B);
  Eigen::MatrixXd INV = llt_denom.solve(Eigen::MatrixXd::Identity(p, p));

  Eigen::MatrixXd SE(X.cols(), Y.cols());
  for (int i = 0; i < SE.cols(); ++i)
  {
    SE.col(i) = Eigen::MatrixXd(p, p).setZero().selfadjointView<Eigen::Upper>().rankUpdate(INV.selfadjointView<Eigen::Upper>() * Xt_Z_Uinv * (Z_Uinv.transpose() * U.col(i).cwiseAbs().asDiagonal())).diagonal().cwiseSqrt();
  }

  return List::create(Named("BETA") = B,
                      Named("SE") = SE);
}

// [[Rcpp::export]]
List iv_cpp_cluster(const Eigen::Map<Eigen::MatrixXd> &X, const Eigen::Map<Eigen::MatrixXd> &Z, const Eigen::Map<Eigen::MatrixXd> &Y, const NumericVector &C)
{

  int p = X.cols();

  Eigen::LLT<Eigen::MatrixXd> llt_ZtZ(Eigen::MatrixXd(p, p).setZero().selfadjointView<Eigen::Upper>().rankUpdate(Z.adjoint()));
  // Eigen::MatrixXd PZ = Z * llt.solve(Eigen::MatrixXd::Identity(p, p)).selfadjointView<Eigen::Upper>() * Z.transpose();
  Eigen::MatrixXd Uinv = llt_ZtZ.matrixU().solve(Eigen::MatrixXd::Identity(p, p));
  Eigen::MatrixXd Z_Uinv = Z * Uinv.triangularView<Eigen::Upper>();
  Eigen::MatrixXd Xt_Z_Uinv = X.transpose() * Z_Uinv;
  Eigen::MatrixXd Xt_PZ_X = Eigen::MatrixXd(p, p).setZero().selfadjointView<Eigen::Upper>().rankUpdate(Xt_Z_Uinv);

  Eigen::LLT<Eigen::MatrixXd> llt_denom((Xt_PZ_X).selfadjointView<Eigen::Upper>());
  Eigen::MatrixXd B = llt_denom.solve(Xt_Z_Uinv * (Z_Uinv.transpose() * Y));
  Eigen::MatrixXd U = (Y - X * B);
  Eigen::MatrixXd INV = llt_denom.solve(Eigen::MatrixXd::Identity(p, p));
  NumericVector C_u = unique(C);

  Eigen::MatrixXd SE(X.cols(), Y.cols());
  for (int i = 0; i < SE.cols(); ++i)
  {
    NumericVector U_i = wrap(U.col(i));
    Eigen::MatrixXd Sigma_i = Eigen::MatrixXd(U_i.size(), U_i.size()).setZero().selfadjointView<Eigen::Upper>();

    for (int c = 0; c < C_u.size(); ++c)
    {
      NumericVector U_i_c = clone(U_i);

      //subset U_i for cluster c
      NumericVector c_vec(C.size());
      c_vec.fill(C_u[c]);
      LogicalVector mask = (C != c_vec);
      U_i_c[mask] = 0;

      Sigma_i.selfadjointView<Eigen::Upper>().rankUpdate(as<Eigen::Map<Eigen::VectorXd>>(U_i_c));
    }

    SE.col(i) = Eigen::MatrixXd(p, p).setZero().selfadjointView<Eigen::Upper>().rankUpdate(INV.selfadjointView<Eigen::Upper>() * Xt_Z_Uinv * (Z_Uinv.transpose() * Sigma_i.llt().matrixL())).diagonal().cwiseSqrt();
  }

  return List::create(Named("BETA") = B,
                      Named("SE") = SE);
}


// [[Rcpp::export]]
Eigen::MatrixXd RIDGE_K(const Eigen::Map<Eigen::MatrixXd> &X, const Eigen::Map<Eigen::MatrixXd> &Y, const int &K)
{
  // Eigen::MatrixXd Y = Standardize(Y_input);
  // Eigen::MatrixXd X = Standardize(X_input);

  int p = X.cols();
  Eigen::MatrixXd XtX = AtA(X);
  
  Eigen::LLT<Eigen::MatrixXd> denom_llt((XtX + (Eigen::MatrixXd::Identity(p, p) * K)).selfadjointView<Eigen::Upper>());
  Eigen::MatrixXd B = denom_llt.solve(X.transpose() * Y);

  return B;
}

// List RIDGE_K(const Eigen::Map<Eigen::MatrixXd> &X_input, const Eigen::Map<Eigen::MatrixXd> &Y_input, const int &K)
// {

//   Eigen::MatrixXd Y = Standardize(Y_input);
//   Eigen::MatrixXd X = Standardize(X_input);
//   Eigen::MatrixXd XtX = X.transpose() * X;
//   int p = X.cols();

//   Eigen::LLT<Eigen::MatrixXd> llt;
//   llt.compute(XtX + Eigen::MatrixXd::Identity(p, p) * K);
//   Eigen::VectorXd B = llt.solve(X.transpose() * Y);
//   Eigen::MatrixXd INV = llt.solve(Eigen::MatrixXd::Identity(p, p));
//   double sigma = (Y - X * B).squaredNorm() / (Y.rows() - p);
//   Eigen::VectorXd SE = sqrt(sigma) * (INV * XtX * INV).diagonal().cwiseSqrt();

//   return List::create(Named("BETA") = B,
//                       Named("SE") = SE,
//                       Named("R2") = 1 - sigma * (Y.rows() - p) / (Y.rows() - 1));
// }

Eigen::ArrayXd Dplus(const Eigen::ArrayXd &d)
{
  Eigen::ArrayXd di(d.size());
  for (int j = 0; j < d.size(); ++j)  di[j] = 1 / d[j];
  return di;
}

// [[Rcpp::export]]
Eigen::MatrixXd RIDGE_multi_K(const Eigen::Map<Eigen::MatrixXd> &X, const Eigen::Map<Eigen::MatrixXd> &Y, const Eigen::Map<Eigen::VectorXd> &K)
{

      Eigen::BDCSVD<Eigen::MatrixXd>
          UDV(X, Eigen::ComputeThinU | Eigen::ComputeThinV);

  int p = X.cols();

  // this needs to be computed once.
  Eigen::MatrixXd DUtY = UDV.singularValues().matrix().asDiagonal() * (UDV.matrixU().transpose() * Y);
  // Eigen::MatrixXd VDsq_p = UDV.matrixV() * (Dplus(UDV.singularValues())).pow(2).matrix().asDiagonal();

  Eigen::MatrixXd B(p, K.size());  
  for (int i = 0; i < K.size(); ++i)
    {
      Eigen::MatrixXd VDsq_p = UDV.matrixV() * (Dplus(UDV.singularValues().array().pow(2) + K[i])).matrix().asDiagonal();
      B.col(i) = VDsq_p.matrix() * DUtY;
      // B.col(i) = (VDsq_p.array()/K[i]).matrix() * DUtY;
    }

    return B;
}

// [[Rcpp::export]]
Eigen::MatrixXd RIDGE_multi_K_llt(const Eigen::Map<Eigen::MatrixXd> &X, const Eigen::Map<Eigen::MatrixXd> &Y, const Eigen::Map<Eigen::VectorXd> &K)
{
  // Eigen::MatrixXd Y = Standardize(Y_input);
  // Eigen::MatrixXd X = Standardize(X_input);

  int p = X.cols();
  Eigen::MatrixXd XtX = AtA(X);
  Eigen::MatrixXd XtY = X.transpose() * Y;

  Eigen::MatrixXd B(p, K.size());
  for (int i = 0; i < K.size(); ++i)
  {
    Eigen::LLT<Eigen::MatrixXd> denom_llt((XtX + (Eigen::MatrixXd::Identity(p, p) * K[i])).selfadjointView<Eigen::Upper>());
    B.col(i) = denom_llt.solve(XtY);
  }
  return B;
}

// [[Rcpp::export]]
List SVD_cpp(const Eigen::Map<Eigen::MatrixXd> &X)
{
  Eigen::BDCSVD<Eigen::MatrixXd> UDV(X, Eigen::ComputeThinU | Eigen::ComputeThinV);

  return List::create(Named("d") = UDV.singularValues(),
                      Named("U") = UDV.matrixU(),
                      Named("V") = UDV.matrixV());
}

  // [[Rcpp::export]]
  List RIDGE_IV_K(const Eigen::Map<Eigen::MatrixXd> &X_input, const Eigen::Map<Eigen::MatrixXd> &Z_input, const Eigen::Map<Eigen::MatrixXd> &Y_input, const int &K)
  {

    Eigen::MatrixXd Y = Standardize(Y_input);
    Eigen::MatrixXd X = Standardize(X_input);
    Eigen::MatrixXd Z = Standardize(Z_input);
    int p = X.cols();

    Eigen::LLT<Eigen::MatrixXd> llt(Eigen::MatrixXd(p, p).setZero().selfadjointView<Eigen::Upper>().rankUpdate(Z.adjoint()));
    // Eigen::MatrixXd PZ = Z * llt.solve(Eigen::MatrixXd::Identity(p, p)).selfadjointView<Eigen::Upper>() * Z.transpose();
    Eigen::MatrixXd Uinv = llt.matrixU().solve(Eigen::MatrixXd::Identity(p, p));
    Eigen::MatrixXd Z_Univ = Z * Uinv.triangularView<Eigen::Upper>();
    Eigen::MatrixXd Xt_Z_Univ = X.transpose() * Z_Univ;
    Eigen::MatrixXd Xt_PZ_X = Eigen::MatrixXd(p, p).setZero().selfadjointView<Eigen::Upper>().rankUpdate(Xt_Z_Univ);

    Eigen::LLT<Eigen::MatrixXd> llt_denom((Xt_PZ_X + Eigen::MatrixXd::Identity(p, p) * K).selfadjointView<Eigen::Upper>());
    Eigen::VectorXd B = llt_denom.solve(Xt_Z_Univ * (Z_Univ.transpose() * Y));
    Eigen::MatrixXd INV = llt_denom.solve(Eigen::MatrixXd::Identity(p, p));
    double sigma = (Y - X * B).squaredNorm() / (Y.rows() - p);
    Eigen::VectorXd SE = sqrt(sigma) * (INV.selfadjointView<Eigen::Upper>() * Xt_PZ_X * INV.selfadjointView<Eigen::Upper>()).diagonal().cwiseSqrt();

    // Eigen::MatrixXd output(2, B.rows());
    // output << B.transpose(), SE.transpose();
    // return output;
    return List::create(Named("BETA") = B,
                        Named("SE") = SE,
                        Named("R2") = 1 - sigma * (Y.rows() - p) / (Y.rows() - 1));
  }

  // // [[Rcpp::export]]
  // Eigen::MatrixXd RIDGE_IV_K_test(const Eigen::Map<Eigen::MatrixXd>& X_input, const Eigen::Map<Eigen::MatrixXd>& Z_input, const Eigen::Map<Eigen::MatrixXd>& Y_input,  const int& K) {

  //   Eigen::MatrixXd Y = Standardize(Y_input);
  //   Eigen::MatrixXd X = Standardize(X_input);
  //   Eigen::MatrixXd Z = Standardize(Z_input);

  //   Eigen::LLT<Eigen::MatrixXd> llt;
  //   llt.compute(Z.transpose() * Z);
  //   Eigen::MatrixXd PZ = Z * llt.solve(Eigen::MatrixXd::Identity(X.cols(), X.cols())) * Z.transpose();
  //   Eigen::MatrixXd XPZX = X.transpose() * PZ * X;
  //   Eigen::MatrixXd XPZY = X.transpose() * PZ * Y;

  //   llt.compute(XPZX + Eigen::MatrixXd::Identity(X.cols(), X.cols()) * K);
  //   Eigen::VectorXd B = llt.solve(XPZY);
  //   Eigen::MatrixXd INV = llt.solve(Eigen::MatrixXd::Identity(X.cols(), X.cols()));
  //   double sigma = (Y - X * B).squaredNorm() / (Y.rows() - X.cols());
  //   Eigen::VectorXd SE = sqrt(sigma) * (INV * XPZX * INV).diagonal().cwiseSqrt();

  //   Eigen::MatrixXd output(2, B.rows());
  //   output << B.transpose(), SE.transpose();
  //   return output;
  // }

  // // [[Rcpp::export]]
  // Eigen::MatrixXd RIDGE_LW(const Eigen::Map<Eigen::MatrixXd>& X_input, const Eigen::Map<Eigen::MatrixXd>& Y_input) {

  // Eigen::MatrixXd Y = Standardize(Y_input);
  // Eigen::MatrixXd X = Standardize(X_input);
  // Eigen::MatrixXd XtX = X.transpose()*X;

  //   // determine K
  //   Eigen::VectorXd B_ols = XtX.llt().solve(X.transpose()*Y);
  //   double sigma_ols = (Y - X*B_ols).squaredNorm() / (Y.rows() - X.cols());
  //   double K = X.cols() * sigma_ols / (B_ols.array() * XtX.eigenvalues().cwiseSqrt().array()).matrix().squaredNorm();

  //   std::cout << "-- parameter lambda: " << K << std::endl ;

  // Eigen::LLT<Eigen::MatrixXd> llt;
  // llt.compute(XtX + Eigen::MatrixXd::Identity(X.cols(), X.cols()) * K );
  // Eigen::VectorXd B = llt.solve(X.transpose()*Y);
  // Eigen::MatrixXd INV = llt.solve( Eigen::MatrixXd::Identity(X.cols(), X.cols()) );
  // double sigma = (Y - X*B).squaredNorm() / (Y.rows() - X.cols());
  // Eigen::VectorXd SE = (sigma * INV * XtX * INV).diagonal().cwiseSqrt();

  // Eigen::MatrixXd output(2, B.rows());
  // output << B.transpose(), SE.transpose();
  // return output;
  // }

  // // [[Rcpp::export]]
  // Eigen::MatrixXd RIDGE_HK(const Eigen::Map<Eigen::MatrixXd>& X_input, const Eigen::Map<Eigen::MatrixXd>& Y_input) {

  // Eigen::MatrixXd Y = Standardize(Y_input);
  // Eigen::MatrixXd X = Standardize(X_input);
  // Eigen::MatrixXd XtX = X.transpose()*X;

  //   // determine K
  //   Eigen::VectorXd B_ols = XtX.llt().solve(X.transpose()*Y);
  //   double sigma_ols = (Y - X*B_ols).squaredNorm() / (Y.rows() - X.cols());
  //   double K = X.cols() * sigma_ols / B_ols.squaredNorm();

  //   std::cout << "-- parameter lambda: " << K << std::endl ;

  // Eigen::LLT<Eigen::MatrixXd> llt;
  // llt.compute(XtX + Eigen::MatrixXd::Identity(X.cols(), X.cols()) * K );
  // Eigen::VectorXd B = llt.solve(X.transpose()*Y);
  // Eigen::MatrixXd INV = llt.solve( Eigen::MatrixXd::Identity(X.cols(), X.cols()) );
  // double sigma = (Y - X*B).squaredNorm() / (Y.rows() - X.cols());
  // Eigen::VectorXd SE = (sigma * INV * XtX * INV).diagonal().cwiseSqrt();

  // Eigen::MatrixXd output(2, B.rows());
  // output << B.transpose(), SE.transpose();
  // return output;
  // }

  // ArrayXd E = (stdY - Q*QtY).array();
  // double SSR = E.square().sum();
  // ArrayXd E_r = (stdY - Q_r*(Q_r.transpose()*stdY)).array();
  // double SSR_r = E_r.square().sum();

  // VectorXd chol_solve(const Map<MatrixXd> &X, const Map<MatrixXd> &Q, const Map<VectorXd> &Y) {

  // VectorXd XtY = X.transpose()*Y;
  // VectorXd B = Q.llt().solve(XtY);
  // return B;
  // }

  // VectorXd reg_F(const Map<MatrixXd> &R, const Map<MatrixXd> &Q, Map<MatrixXd> &Q_r, const Map<VectorXd> &Y) {

  // // int nrow = Y.rows();
  // // int ncol = Y.cols();
  // // int p    = R.rows();

  // VectorXd stdY = Standardize(Y);
  // VectorXd QtY = Q.transpose()*stdY;
  // VectorXd B = R.triangularView<Upper>().solve(QtY);
  // double SSR = (stdY - Q*QtY).squaredNorm();
  // double SSR_r = (stdY - Q_r*(Q_r.transpose()*stdY)).squaredNorm();

  // VectorXd output(B.rows() + 2);
  // output << B, SSR, SSR_r;
  // return output;
  // }

  // MatrixXd reg_F(const Map<MatrixXd> &R, const Map<MatrixXd> &Q, Map<MatrixXd> &Q_r, const Map<MatrixXd> &Y) {

  // MatrixXd stdY = Standardize(Y);
  // MatrixXd QtY = Q.transpose()*stdY;
  // MatrixXd B = R.triangularView<Upper>().solve(QtY);
  // VectorXd SSR = (stdY - Q*QtY).colwise().squaredNorm();
  // VectorXd SSR_r = (stdY - Q_r*(Q_r.transpose()*stdY)).colwise().squaredNorm();

  // MatrixXd output(B.rows() + 2, B.cols());
  // output << B, SSR.transpose(), SSR_r.transpose();
  // return output;
  // }