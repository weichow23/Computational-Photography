// sparse.h
#ifndef _SPARSE_H
#define _SPARSE_H

#include <iostream>
#include <vector>
/* include other libraries if you need */

typedef std::vector<double> Vecd;
typedef std::vector<int> Veci;


class Mnode{
    public:
        int i; // Row id
        int j; // Col id
        double v;   // Element value
        Mnode* right;
        Mnode* down;
        Mnode(int i = 0, int j = 0, double v = 0, Mnode* right = nullptr, Mnode* down = nullptr)
        : i(i), j(j), v(v), right(right), down(down) {}
};

/**
 * @brief Sparse Matrix Class.
 * Assume: 
 * 1. number of nonzero elements less than int
 * 2. every element is under the precision of 1e-10
 *    (if fabs(x) < 1e-10, then x is considered as 0)
*/
class Sparse {
    public:
        const double epsilon = 1e-10;   //  precision
        int RowNum, ColNum, NonzeronNum;
        Mnode* data;

        /* define constructors and destructor if you need */
        Sparse() : RowNum(0), ColNum(0), NonzeronNum(0), data(nullptr) {}

        Sparse(int md, int nd) :RowNum(md), ColNum(nd), NonzeronNum(0){
            data = new Mnode [RowNum];
        }
        
        ~Sparse() {
            clear();
            delete[] data;
        }

        void clear() {
            for (int i = 0; i < RowNum; ++i) {
                Mnode* currentNode = data[i].right;
                while (currentNode != nullptr) {
                    Mnode* toDelete = currentNode;
                    currentNode = currentNode->right;
                    delete toDelete;
                }
                data[i].right = nullptr; // Reset the row header's next pointer
            }
            NonzeronNum = 0; // Reset the count of non-zero elements
        }

        int getRowDimension() {
            return RowNum;
        }
        int getColDimension() {
            return ColNum;
        }

        /**
         * @brief read the element at Matrix[row][column]
         * 
         * @param row row index, starts from 0, like 0, 1, 2, 3 ...
         * @param col column index, starts from 0, like 0, 1, 2, 3 ...
        */
        // double at(int row, int col) const;
        int at(int row, int col) {
            if (row < 0 || row >= RowNum || col < 0 || col >= ColNum) {
                cout << "out of range" << endl;
                return 0;
            }
            Mnode* line = &data[row];
            Mnode* value = GetElement(line, col);
            if (value == nullptr) {
                return 0;
            }
            return value->v;
        }
        
        /**
         * @brief insert / modify the element at Matrix[row][column]
         * 
         * @param val value to insert / modify
         * @param row row indices, starts from 0, like 0, 1, 2, 3 ...
         * @param col column indices, starts from 0, like 0, 1, 2, 3 ...
        */
        // void insert(double val, int row, int col);
        Mnode* insert(double val, int row, int col) {
            if (NonzeronNum >= RowNum * ColNum || row > RowNum || col > ColNum){
                return nullptr;
            }
            Mnode* p = &data[row];
            while (p) {
                if (p->j == col) {
                    if (p->v == 0 && val != 0) {
                        NonzeronNum++;
                    }
                    p->v = val;
                    return p;
                }
                if (p->j < col && p->right && p->right->j > col) {
                    Mnode* newele = new Mnode(row, col, val);
                    NonzeronNum++;
                    newele->right = p->right;
                    p->right = newele;
                    return newele;
                }
                if (p->right == nullptr) {
                    Mnode* newele = new Mnode(row, col, val);
                    NonzeronNum++;
                    p->right = newele;
                    return newele;
                }
                p = p->right;
            }
        }

        /**
         * @brief initialize the sparse matrix with 3 vectors (equal lengths)
         * 
         * @param rows for row indices
         * @param cols for column indices
         * @param vals for values. vals[i] = Matrix[rows[i]][cols[i]]
        */
        // void initializeFromVector(const Veci& rows, const Veci& cols, const Vecd& vals);
        void initializeFromVector(Veci rows, Veci cols, Vecd vals) {
            delete data;
            NonzeronNum = 0;
            auto R = max_element(rows.begin(), rows.end());
            auto C = max_element(cols.begin(), cols.end());
            RowNum = *R + 1;
            ColNum = *C + 1;
            data = new Mnode[RowNum];
            Mnode* p = data;

            for (int i = 0; i < rows.size(); i++) {
                int valuei = rows[i];
                int valuej = cols[i];
                double value = vals[i];
                insert(value, valuei, valuej);
            }
        }

        /* define other public member variables and functions if you need */
        Mnode* GetElement(Mnode* line, int j){
            Mnode* p = line;
            while (p){
                if (p->j == j) {
                    return p;
                }
                p = p->right;
            }
            return nullptr;
        }

        void Print() {
            cout << "Matrix as follow: " << ":" << endl;
            for (int i = 0; i < RowNum; i++) {
                for (int j = 0; j < ColNum; j++) {
                    cout << at(i, j) << "  ";
                }
                cout << endl;
            }
        }
};
#endif