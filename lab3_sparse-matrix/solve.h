#ifndef _HW3_SOLVE_H
#define _HW3_SOLVE_H

#include "sparse.h"
#include <cmath>

/**
 * @brief solve Ax=b with Gauss-Seidel method
 * 
 * @param A coefficient matrix
 * @param b label vector (just a nickname)
 * @param error when infinity norm of |x_{k+1} - x_k| < (or <=) error, stop iteration
 * @return x: Vecd, solution of Ax=b
 */
//Vecd Gauss_Seidel(const Sparse& A, const Vecd& b, double error);
Vecd Gauss_Seidel(Sparse& A, const Vecd& b, double error) {
    int rows = A.getRowDimension();
    Vecd x(rows, 0.0);
    Vecd x_old(rows, std::numeric_limits<double>::max()); 
    double maxDiff = std::numeric_limits<double>::max(); 

    while (maxDiff > error) {
        maxDiff = 0.0; 
        for (int i = 0; i < rows; i++) {
            double sum = 0.0;
            for (int j = 0; j < rows; j++) {
                if (j != i) {
                    sum += A.at(i, j) * x[j];
                }
            }

            if (std::abs(A.at(i, i)) < A.epsilon) {
                std::cerr << "Diagonal element too close to zero at row " << i << ". Cannot proceed." << std::endl;
                return Vecd();
            }

            double temp = x[i];
            x[i] = (b[i] - sum) / A.at(i, i);
            maxDiff = std::max(maxDiff, std::abs(x[i] - temp));
        }
    }
    return x;
}


Vecd multiplyVector(Sparse& A, Vecd& x) {
    Vecd result(x.size(), 0.0);
    for (int i = 0; i < A.getRowDimension(); ++i) {
        for (int j = 0; j < A.getColDimension(); ++j) {
            result[i] += A.at(i, j) * x[j];
        }
    }
    return result;
}

Vecd subVector(const Vecd& x, const Vecd& y) {
    Vecd result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] - y[i];
    }
    return result;
}

Vecd addVector(const Vecd& x, const Vecd& y) {
    Vecd result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] + y[i];
    }
    return result;
}

Vecd scalarMultiplyVector(double scalar, const Vecd& y) {
    Vecd result(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        result[i] = scalar * y[i];
    }
    return result;
}

double dotProduct(const Vecd& x, const Vecd& y) {
    double result = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        result += x[i] * y[i];
    }
    return result;
}

double norm(const Vecd& x) {
    return std::sqrt(dotProduct(x, x));
}

/**
 * @brief solve Ax=b with conjugate gradient method
 * 
 * @param A coefficient matrix
 * @param b label vector (just a nickname)
 * @param error when ||r||_2^2 < (or <=) error, stop iteration. r = b - Ax is the residual
 * @param kmax max iterations
 * @return x: Vecd, solution of Ax=b 
 */
// Vecd Conjugate_Gradient(const Sparse& A, const Vecd& b, double error, int kmax);
Vecd Conjugate_Gradient(Sparse& A, const Vecd& b, double error, int kmax) {
    Vecd x(b.size(), 0.0); // Start with an initial guess of zeros
    Vecd r = subVector(b, multiplyVector(A, x));
    Vecd p = r;
    double rsold = dotProduct(r, r);

    for (int k = 0; k < kmax; ++k) {
        Vecd Ap = multiplyVector(A, p);
        double alpha = rsold / dotProduct(p, Ap);
        x = addVector(x, scalarMultiplyVector(alpha, p));
        r = subVector(r, scalarMultiplyVector(alpha, Ap));
        double rsnew = dotProduct(r, r);
        if (std::sqrt(rsnew) < error) {
            break; // Convergence check
        }
        p = addVector(r, scalarMultiplyVector(rsnew / rsold, p));
        rsold = rsnew;
    }
    return x;
}
#endif