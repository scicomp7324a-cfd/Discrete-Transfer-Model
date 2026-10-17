# pragma once

#include <iostream>
#include <vector>
#include <stdexcept>

#include "linear_algebra/SparseLinearSystem.hpp"
#include "linear_algebra/GaussSeidelSmoother.hpp"

using namespace std ;

/* *******************************************************************
    GaussSeidelSolver.hpp - This class contains the solve function that 
    runs the Gauss Seidel smoother. It also controls the number of 
    sweeps per check and the maximum number of smoother iterations.
******************************************************************* */

class GaussSeidelSolver: public GaussSeidelSmoother{

private:
        
    SparseLinearSystem& sys_ ;
    int maxIter = 200;
    int sweepsPerCheck = 10 ;
    double resInitial = 0.0 ;
    double resFinal = 0.0 ;

public:

    GaussSeidelSolver(SparseLinearSystem& sys)
    : GaussSeidelSmoother(sys),
      sys_(sys)
    {}

    void solve(double relTol, bool verbose){

        resInitial = 0.0 ;
        resFinal = 0.0 ;

        int numSweepsCompleted = 0 ;

        sys_.calculateResidual() ;
        resInitial = sys_.r().twoNorm() ;

        if(resInitial == 0.0){
            resFinal = 0.0;

            if(verbose){
                cout << "Initial Residual: " << resInitial
                     << ", Final Residual: " << resFinal << endl ;
            }

            return;
        }

        while(numSweepsCompleted < maxIter){

            smooth(sweepsPerCheck) ;

            numSweepsCompleted += sweepsPerCheck ;

            sys_.calculateResidual() ;
            resFinal = sys_.r().twoNorm() ;

            double ratio = resFinal / resInitial ;

            if(ratio < relTol){
                break ;
            }
        }

        if(verbose){
            cout << "Initial Residual: " << resInitial
                 << ", Final Residual: " << resFinal << endl ;
        }     
    }
};