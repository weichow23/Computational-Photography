#ifndef HW3_GN_34804D67
#define HW3_GN_34804D67
#include <opencv2/opencv.hpp>
#include <math.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
using namespace std;
using namespace cv;

#define POW(x) (pow(x, 2))
#define POW_n3(x) (1.0/pow(x, 3))


struct GaussNewtonParams {
	GaussNewtonParams() :
		exact_line_search(false),
		gradient_tolerance(1e-5),
		residual_tolerance(1e-5),
		max_iter(1000),
		verbose(false)
	{}
	bool exact_line_search; // ʹ�þ�ȷ�����������ǽ�����������
	double gradient_tolerance; // �ݶ���ֵ����ǰ�ݶ�С�������ֵʱֹͣ����
	double residual_tolerance; // ������ֵ����ǰ����С�������ֵʱֹͣ����
	int max_iter; // ����������
	bool verbose; // �Ƿ��ӡÿ����������Ϣ
};

struct GaussNewtonReport {
	enum StopType {
		STOP_GRAD_TOL,       // �ݶȴﵽ��ֵ
		STOP_RESIDUAL_TOL,   // ����ﵽ��ֵ
		STOP_NO_CONVERGE,    // ������
		STOP_NUMERIC_FAILURE // ������ֵ����
	};
	StopType stop_type; // �Ż���ֹ��ԭ��
	double n_iter;      // ��������
};

class ResidualFunction {
public:
	virtual int nR() const = 0;
	virtual int nX() const = 0;
	virtual void eval(double *R, double *J, double *X) = 0;
};

class GaussNewtonSolver {
public:
	virtual double solve(
		ResidualFunction *f, // Ŀ�꺯��
		double *X,           // ������Ϊ��ֵ�������Ϊ���
		GaussNewtonParams param = GaussNewtonParams(), // �Ż�����
		GaussNewtonReport *report = nullptr // �Ż��������
	) = 0;
};

// ---------------------- ����Ϊ���ӵĴ��� ----------------------
class Optimizer : public ResidualFunction {
public :
	double *x,*y,*z;
	int dimension = 3;
	int size;

	Optimizer() {
        readDataFromFile("ellipse753.txt");
    }

    ~Optimizer() {
        delete[] x; delete[] y; delete[] z;
    }

    void readDataFromFile(const string& filename) {
        ifstream file(filename);
        string line;
        vector<vector<double>> points;

        while (getline(file, line)) {
            stringstream ss(line);
            double val;
            vector<double> point;
            while (ss >> val) {
                point.push_back(val);
            }
            if(point.size() == 3)
                points.push_back(point);
        }

        size = points.size();
        x = new double[size];
        y = new double[size];
        z = new double[size];
        
        for (int i = 0; i < size; i++) {
            x[i] = points[i][0];
            y[i] = points[i][1];
            z[i] = points[i][2];
        }
    }

	double cal_value(double *coefficient, int i) {
		double A = coefficient[0];
		double B = coefficient[1];
		double C = coefficient[2];
		double result = 1 - 1.0 / POW(A)*POW(x[i]) - 1.0 / POW(B)*POW(y[i]) - 1.0 / POW(C)*POW(z[i]); // 目标函数
		return result;
	}


	virtual int nR() const {
		return size;
	}

	virtual int nX() const {
		return dimension;
	}

	virtual void eval(double *R, double *J, double *coefficient) {
		for (int i = 0; i < size; i++) {
			R[i] = cal_value(coefficient, i);
			J[i * 3 + 0] = -2 * POW(x[i])*POW_n3(coefficient[0]);
			J[i * 3 + 1] = -2 * POW(y[i])*POW_n3(coefficient[1]);
			J[i * 3 + 2] = -2 * POW(z[i])*POW_n3(coefficient[2]);
		}
	}
};

class Sover3790 : public GaussNewtonSolver {
	public:
		virtual double solve(
			ResidualFunction *f,
			double *X, 
			GaussNewtonParams param = GaussNewtonParams(),
			GaussNewtonReport *report = nullptr
		) override {
			double *x = X;
			int n = 0;
			double step = 1;
			int nR = f->nR();
			int nX = f->nX();
			double *J = new double[nR*nX];
			double *R = new double[nR];

			while (n < param.max_iter) {
				f->eval(R, J, x);
				Mat mat_R(nR, 1,CV_64FC1, R);
				Mat mat_J(nR, nX, CV_64FC1, J);
				Mat mat_Delta(nX, 1, CV_64FC1);
				cv::solve(mat_J, mat_R, mat_Delta, DECOMP_SVD);

				double max_R = getMaxAbs(mat_R);
				double max_mat_Delta = getMaxAbs(mat_Delta);

				if (checkConvergence(max_R, max_mat_Delta, param, report, n)) 
					return 0; // Converged
					
				updateX(x, mat_Delta, step, nX);
				n++;
			}

			setNoConverge(report, n);
			return 1; // No Convergence
		}

	private:
		double getMaxAbs(const Mat &mat) {
			double max_val = -1;
			for (int i = 0; i < mat.rows; i++) {
				double val = abs(mat.at<double>(i, 0));
				if (val > max_val) {
					max_val = val;
				}
			}
			return max_val;
		}

		bool checkConvergence(double max_R, double max_mat_Delta, GaussNewtonParams &param, GaussNewtonReport *report, int n) {
			if (max_R <= param.residual_tolerance || max_mat_Delta <= param.gradient_tolerance) {
				report->stop_type = (max_R <= param.residual_tolerance) ? report->STOP_RESIDUAL_TOL : report->STOP_RESIDUAL_TOL;
				report->n_iter = n;
				return true;
			}
			return false;
		}

		void updateX(double *x, const Mat &mat_Delta, double step, int nX) {
			for (int i = 0; i < nX; i++) {
				x[i] += step * mat_Delta.at<double>(i, 0);
			}
		}

		void setNoConverge(GaussNewtonReport *report, int n) {
			report->stop_type = report->STOP_NO_CONVERGE;
			report->n_iter = n;
		}
};


#endif /* HW3_GN_34804D67 */