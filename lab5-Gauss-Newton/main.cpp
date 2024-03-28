#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <vector>
#include "hw3_gn.h"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    vector<double> Coefficient(3, 3.0); // Initialize all coefficients to 3.0
    unique_ptr<ResidualFunction> s = make_unique<Optimizer>();
    GaussNewtonParams param;
    GaussNewtonReport report;
    unique_ptr<Sover3790> mysolver = make_unique<Sover3790>();
    
    mysolver->solve(s.get(), Coefficient.data(), param, &report);
    cout << "Report:" << endl;
    cout << "Num of iteration: " << report.n_iter << endl;
    
    char co[3] = {'A', 'B', 'C'};
    for (int i = 0; i < 3; i++) {
        cout << co[i] << " " << Coefficient[i] << endl;
    }

    switch (report.stop_type) {
        case GaussNewtonReport::STOP_RESIDUAL_TOL:
            cout << "[Stop]: The remainder reaches the threshold" << endl; break;
        case GaussNewtonReport::STOP_GRAD_TOL:
            cout << "[Stop]: Gradient reaches threshold" << endl; break;
        case GaussNewtonReport::STOP_NO_CONVERGE:
            cout << "[Stop]: Does not converge" << endl; break;
        default:
            cout << "[Stop]: Error!" << endl; break;
    }
    getchar();
}
