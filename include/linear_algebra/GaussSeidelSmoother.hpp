# pragma once

#include <iostream>
#include <vector>
#include <stdexcept>

#include "linear_algebra/SparseLinearSystem.hpp"
#include "linear_algebra/SparseMatrix.hpp"

using namespace std ;

/* *******************************************************************
    GaussSeidelSolver.hpp - COO Gauss-Seidel smoother
******************************************************************* */

class GaussSeidelSmoother{

private:

    SparseLinearSystem& sys_ ;

public:

    explicit GaussSeidelSmoother(SparseLinearSystem& sys)
    : sys_(sys)
    {}

    void smooth(int nSweeps){

        int numEq = sys_.A().n();

        for(int sweep = 0; sweep < nSweeps; ++sweep){

            for(int i = 0; i < numEq; ++i){

                double sum = 0.0;
                double diag = 0.0;

                for(const coo& elem : sys_.A().entries()){

                    if(elem.row != i){
                        continue;
                    }

                    if(elem.col == i){
                        diag = elem.val;
                    }
                    else{
                        sum += elem.val * sys_.X()(elem.col);
                    }
                }

                if(diag == 0.0){
                    throw runtime_error("Zero diagonal found in Gauss-Seidel smoother");
                }

                sys_.X()(i) = (sys_.b()(i) - sum) / diag;
            }
        }
    }
};