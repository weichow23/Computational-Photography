using namespace std;
#include "sparse.h"
#include "solve.h"

void test_solve(Sparse& A, Vecd& b){
	Vecd X = Gauss_Seidel(A, b, 0.000000001);
	cout << "Gauss_Seidel:" << endl;
	for (int i = 0; i < A.ColNum; i++)
		cout << X[i] << '\t';
	
	Vecd XC = Conjugate_Gradient(A, b, 0.000000001, 100000);
	cout << "\nConjugate_Gradient:" << endl;
	for (int i = 0; i < A.ColNum; i++)
		cout << XC[i] << '\t';
	cout << endl;
}

int main() {
	cout << "-------------------------- Sparse Test ---------------------------" << endl;
	Sparse s(2 ,2);
	s.insert(1, 0, 0);
	cout << "insert 1 in (0,0)" << endl;
	cout << "infer (0,0): " << s.at(0, 0) << endl;
	cout << "infer (1,1): " << s.at(1, 1) << endl;
	s.insert(2, 1, 1);
	cout << "insert 2 in (1,1)" << endl;
	cout << "infer (0,0): " << s.at(0, 0) << endl;
	cout << "infer (1,1): " << s.at(1, 1) << endl;
	cout << "-------------------------- initializeFromVector test --------------" << endl;
	Veci a = { 0,1,2 };
	Veci b = { 0,1,2 };
	Vecd c = { 7,8,9 };
	s.initializeFromVector(a, b, c);
	s.Print();
	cout << "--------------------------- Solve Test 1---------------------------" << endl;
	Veci r1 = { 0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3 };
	Veci c1 = { 0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3 };
	Vecd v1 = { 10,-1,2,0,-1,11,-1,3,2,-1,10,-1,0,3,-1,8 };
	s.initializeFromVector(r1, c1, v1);
	s.Print();
	Vecd B1 = { 6.0, 25.0, -11.0, 15.0 }; 
	test_solve(s, B1);
	cout << "--------------------------- Solve Test 2---------------------------" << endl;
	Veci r2 = {0,0,0,1,1,1,2,2,2};
	Veci c2 = {0,1,2,0,1,2,0,1,2};
	Vecd v2 = {2,2,3,2,5,2,3,1,7};
	Vecd B2 = {15.0, 18.0, 26.0}; 
	s.initializeFromVector(r2, c2, v2);
	s.Print();
	test_solve(s, B2);
	getchar(); //阻塞
}