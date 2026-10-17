# pragma once

#include <iostream>
#include <vector>
#include <filesystem>
#include <stdexcept>
#include <fstream>

#include "linear_algebra/SparseMatrix.hpp"
#include "linear_algebra/LinearVector.hpp"

using namespace std ;
namespace fs = filesystem ;


/* *******************************************************************
    SparseLinearSystem.hpp - This class contains sparse linear system 
    storage and residual calculation functions.
******************************************************************* */

class SparseLinearSystem{

private:

    int n_ ;
    SparseMatrix A_ ;
    LinearVector X_ ;
    LinearVector b_ ;
    LinearVector r_ ;

public:

    SparseLinearSystem(int n)
    : n_(n),
      A_(n),
      X_(n),
      b_(n),
      r_(n)
    {}

    int size() const{
        return n_ ;
    }

    SparseMatrix& A(){
        return A_ ;
    }

    const SparseMatrix& A() const{
        return A_ ;
    }

    LinearVector& X(){
        return X_ ;
    }

    const LinearVector& X() const {
        return X_ ;
    }

    LinearVector& b(){
        return b_;
    }

    const LinearVector& b() const {
        return b_ ;
    }

    LinearVector& r(){
        return r_;
    }

    const LinearVector& r() const {
        return r_ ;
    }

    SparseLinearSystem operator*(const double s) const{

        SparseLinearSystem Lr(n_) ;

        Lr.A() = this->A() * s ;
        Lr.b() = this->b() * s ;
        Lr.X() = this->X() ;

        return Lr ;
    }

    LinearVector operator*(const LinearVector& x) const{

        if(x.size() != n_){
            throw runtime_error("System size mismatch");
        }

        return this->A() * x ;
    }

    void calculateResidual(){

        LinearVector Ax = (*this) * this->X() ;

        for(int i = 0; i < n_; ++i){
            this->r()(i) = this->b()(i) - Ax(i) ;
        }
    }

};