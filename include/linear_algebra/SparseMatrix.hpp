#pragma once

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include "linear_algebra/LinearVector.hpp"

using namespace std;

struct coo{
    int row;
    int col;
    double val;
};

class SparseMatrix{

private:

    int n_;
    vector<coo> A_;

public:

    SparseMatrix(int n)
    : n_(n)
    {}

    int n() const {
        return n_;
    }

    int dim() const {
        return n_;
    }

    int nnz() const {
        return static_cast<int>(A_.size());
    }

    vector<coo>& entries(){
        return A_;
    }

    const vector<coo>& entries() const{
        return A_;
    }

    int index(int i, int j) const{
        for(int m = 0; m < static_cast<int>(A_.size()); ++m){
            if(A_[m].row == i && A_[m].col == j){
                return m;
            }
        }
        return -1;
    }

    double operator()(int i, int j) const{
        int id = index(i,j);

        if(id == -1){
            return 0.0;
        }

        return A_[id].val;
    }

    void put(int i, int j, double val){

        if(i < 0 || i >= n_ || j < 0 || j >= n_){
            throw out_of_range("SparseMatrix index out of range");
        }

        int id = index(i,j);

        if(id != -1){
            A_[id].val = val;
        }
        else if(val != 0.0){
            A_.push_back({i,j,val});
        }
    }

    void add(int i, int j, double val){

        if(i < 0 || i >= n_ || j < 0 || j >= n_){
            throw out_of_range("SparseMatrix index out of range");
        }

        if(val == 0.0){
            return;
        }

        int id = index(i,j);

        if(id != -1){
            A_[id].val += val;
        }
        else{
            A_.push_back({i,j,val});
        }
    }

    LinearVector operator*(const LinearVector& X) const{

        LinearVector res(n_);

        for(const coo& elem : A_){
            res(elem.row) += elem.val * X(elem.col);
        }

        return res;
    }

    SparseMatrix operator*(double s) const{

        SparseMatrix result(n_);
        for(const coo& e : A_){
            result.put(e.row, e.col, s * e.val);
        }
        return result;
    }

    void transpose(){

        for(coo& elem : A_){
            std::swap(elem.row, elem.col);
        }
    }

    void rightMultiplyMatrix(const LinearVector& D){

        for(coo& elem : A_){
            elem.val *= D(elem.col);
        }
    }

    void leftMultiplyMatrix(const LinearVector& D){

        for(coo& elem : A_){
            elem.val *= D(elem.row);
        }
    }

    void substractFromIdentityMatrix(){

        for(coo& elem : A_){
            elem.val = -elem.val;
        }

        for(int i = 0; i < n_; ++i){

            int id = index(i,i);

            if(id != -1){
                A_[id].val += 1.0;
            }
            else{
                A_.push_back({i,i,1.0});
            }
        }
    }
};