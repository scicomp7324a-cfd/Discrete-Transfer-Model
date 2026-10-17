# pragma once

#include <iostream>
#include <vector>

using namespace std ;

/* *******************************************************************
    LinearVector.hpp - This class contains all the algebraic 
    operations for linear vector and norm calculation  
    functions.
******************************************************************* */

class LinearVector{

    private:
        int n_ ;
        vector<double> x_ ;
        
    public: 
        // Constructs the linear vector using the given dimension
        LinearVector(int n):n_(n){
            x_.resize(n, 0.0) ;
        }

        // Non-constant operator overload to access an element in the vector 
        double& operator()(int j){
            return x_[j] ;
        }

        // Constant operator overload to access an element in the vector
        const double& operator()(int j) const{
            return x_[j] ;
        }

        // Operator overload to return the whole linear vector
        vector<double>& operator()(){
            return x_ ;
        }

        // Non-constant access to vector data
        double* data(){
            return x_.data();
        }

        // Constant access to vector data
        const double* data() const {
            return x_.data();
        }

        // Operator overload to add two vectors
        LinearVector operator+(const LinearVector& y_) const{

            if (n_ != y_.size()) throw std::runtime_error("Vector size mismatch");

            LinearVector z_(n_) ;
            for(int i=0; i < n_; ++i){
                z_(i) = (*this)(i) + y_(i) ;
            }

            return z_ ;
        }

        // Operator overload to substract two vectors
        LinearVector operator-(const LinearVector& y_) const{

            if (n_ != y_.size()) throw std::runtime_error("Vector size mismatch");

            LinearVector z_(n_) ;
            for(int i=0; i < n_; ++i){
                z_(i) = (*this)(i) - y_(i) ;
            }

            return z_ ;
        }

        // Operator overload to multiply linear vector to scalar
        LinearVector operator*(const double s) const{

            LinearVector z_(n_) ;
            for(int i=0; i < n_; ++i){
                z_(i) = (*this)(i)*s ;
            }
            return z_ ;
        }

        // Operator overload for element wise multiplication for linear vector
        LinearVector operator*(const LinearVector s) const{

            LinearVector z_(n_) ;
            for(int i=0; i < n_; ++i){
                z_(i) = (*this)(i)*s(i) ;
            }
            return z_ ;
        }

        // Function to calculate one norm of the linear vector
        double oneNorm() const{
            double sum = 0;
            for(int i = 0; i < n_; ++i){
                sum += abs((*this)(i)) ;
            }
            return sum ;
        }

        // Function to calculate two norm of the linear vector
        double twoNorm() const{
            double sum = 0;
            for(int i=0; i< n_ ; ++i){
                sum += pow((*this)(i),2) ;
            }     
            sum = sqrt(sum) ;
            return sum ;
        }

        // Function to calculate infinity norm of the linear vector
        double infinityNorm() const{
            double max = 0 ;
            for(int i=0; i< n_; ++i){
                if(abs((*this)(i)) > max){
                    max = abs((*this)(i)) ;
                }
            }
            return max ;
        }

        // Function to return size of linear vector
        int size() const{
            return n_ ;
        }

        // Function to return the maximum absolute value of an element in the linear vector 
        double maxAbsVal(){
            double max = 1e-8 ;
            for(int i=0; i<(*this).size(); ++i){
                if(max < abs((*this)(i))){
                    max = abs((*this)(i)) ;
                }
            }
            return max ;
        }

        // Function to return the minimum absolute value of an element in the linear vector 
        double minAbsVal(){
            double min = 1e8 ;
            for(int i=0; i<(*this).size(); ++i){
                if(min > abs((*this)(i))){
                    min = abs((*this)(i)) ;
                }
            }
            return min ;
        }

        // Function to return the maximum value of an element in the linear vector 
        double maxVal(){
            double max = 1e-8 ;
            for(int i=0; i<(*this).size(); ++i){
                if(max < (*this)(i)){
                    max = (*this)(i) ;
                }
            }
            return max ;
        }

        // Function to return the minimum value of an element in the linear vector 
        double minVal(){
            double min = 1e8 ;
            for(int i=0; i<(*this).size(); ++i){
                if(min > (*this)(i)){
                    min = (*this)(i) ;
                }
            }
            return min ;
        }

    } ;
