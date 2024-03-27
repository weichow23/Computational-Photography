#include <opencv2/opencv.hpp>
#include <math.h>
#include <iostream>

using namespace cv;

void BoxFilter(const Mat& input, Mat& output, int w, int h) {
    // 创建滤波器核
    Mat kern(2 * w + 1, 2 * h + 1, CV_64FC1);
    kern.setTo(1.0 / ((2 * w + 1) * (2 * h + 1)));
    // 进行滤波
    filter2D(input, output, input.depth(), kern);
}

Mat GetGaussianKernel(double sigma) {
    const double PI = acos(-1);
    int newsize = 2 * 5 * sigma + 1;
    Mat gaus = Mat(newsize, newsize, CV_64FC1);

    int center = newsize / 2;
    double sum = 0;
    for (int i = 0; i < newsize; i++) {
        for (int j = 0; j < newsize; j++) {
            gaus.at<double>(i, j) = (1 / (2 * PI * sigma * sigma)) * exp(-((i - center) * (i - center) + (j - center) * (j - center)) / (2 * sigma * sigma));
            sum += gaus.at<double>(i, j);
        }
    }
    for (int i = 0; i < newsize; i++) {
        for (int j = 0; j < newsize; j++) {
            gaus.at<double>(i, j) /= sum;
        }
    }
    return gaus;
}
void GaussianFilter(const Mat& input, Mat& output, double sigma) {
    Mat gaus = GetGaussianKernel(sigma);
    filter2D(input, output, input.depth(), gaus);
}

void MedianFilter(const Mat& input, Mat& output, int w, int h) {
    output = input.clone();
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            // 对每个颜色通道分别处理
            for (int c = 0; c < 3; c++) {
                std::vector<int> neighborhood;
                for (int dy = -h; dy <= h; dy++) {
                    for (int dx = -w; dx <= w; dx++) {
                        int ny = i + dy;
                        int nx = j + dx;
                        if (nx >= 0 && nx < input.cols && ny >= 0 && ny < input.rows) {
                            Vec3b color = input.at<Vec3b>(ny, nx);
                            neighborhood.push_back(color[c]);
                        }
                    }
                }
                std::nth_element(neighborhood.begin(), neighborhood.begin() + neighborhood.size() / 2, neighborhood.end());
                int median = neighborhood[neighborhood.size() / 2];
                output.at<Vec3b>(i, j)[c] = static_cast<uchar>(median);
            }
        }
    }
}


double PI = 3.1415926535;
double Gaussian(double sigma, double x) {
    return exp(-pow(x, 2) / (2 * pow(sigma, 2))) / (sigma * sqrt(2 * PI));
}
double GetDistance(int x1, int y1, int x2, int y2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}
void BilateralFilter(const Mat& input, Mat& output, double sigmaS, double sigmaR) {
    CV_Assert(input.depth() == CV_8U); // 确保图像深度为8位
    output = Mat(input.size(), input.type());
    int w = 5; // 半径窗口大小
    int d = 2 * w + 1; // 窗口直径
    // 对每个像素应用双边滤波
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            double weightSum = 0.0;
            Vec3d Isum(0, 0, 0);

            for (int ii = -w; ii <= w; ii++) {
                for (int jj = -w; jj <= w; jj++) {
                    int x = j + jj;
                    int y = i + ii;

                    if (x >= 0 && x < input.cols && y >= 0 && y < input.rows) {
                        Vec3d diff = input.at<Vec3b>(y, x) - input.at<Vec3b>(i, j);
                        double ds = exp(-0.5 * (ii * ii + jj * jj) / (sigmaS * sigmaS));
                        double dr = exp(-0.5 * (diff.dot(diff)) / (sigmaR * sigmaR));
                        double weight = ds * dr;

                        Isum += weight * input.at<Vec3b>(y, x);
                        weightSum += weight;
                    }
                }
            }
            Vec3b newVal = (1.0 / weightSum) * Isum;
            output.at<Vec3b>(i, j) = newVal;
        }
    }
}



int main(int argc, char* argv[]) {
    std::string inname, outname;
    int w, h;
    double sigma, sigma_s, sigma_r;

    char mode;
    std::cout << "choose a mode \n\b\b b:BoxFilter;\n\b\b g:GaussianFilter;\n\b\b m:MedianFilter;\n\b\b x: BilateralFilter\n please type a char here: ";
    std::cin >> mode;

    char choice;
    std::cout << "Do you want to use default settings? (y/n): ";
    std::cin >> choice;

    // 使用默认参数或者读入参数
    if (choice == 'y' || choice == 'Y') {
        inname = "test4.jpg";
        outname = "output.jpg";
        w = 9;
        h = 9;
        sigma = 9.0;
        sigma_s = 36.0;
        sigma_r = 36.0;
    }
    else if (mode == 'b' || mode == 'B' || mode == 'm' || mode == 'M') {
        std::cout << "Enter input image name(path): ";
        std::cin >> inname;
        std::cout << "Enter output image name(path): ";
        std::cin >> outname;
        std::cout << "Enter width(double): ";
        std::cin >> w;
        std::cout << "Enter height(double): ";
        std::cin >> h;
    }
    else if (mode == 'g' || mode == 'G') {
        std::cout << "Enter input image name(path): ";
        std::cin >> inname;
        std::cout << "Enter output image name(path): ";
        std::cin >> outname;
        std::cout << "Enter sigma(double): ";
        std::cin >> sigma;
    }
    else if (mode == 'x' || mode == 'X') {
        std::cout << "Enter input image name(path): ";
        std::cin >> inname;
        std::cout << "Enter output image name(path): ";
        std::cin >> outname;
        std::cout << "Enter sigma S(double): ";
        std::cin >> sigma_s;
        std::cout << "Enter sigma R(double): ";
        std::cin >> sigma_r;
    }
    else {
        std::cout << "Invalid mode selected." << std::endl;
        return 1;
    }

    // 读取图像
    Mat image = imread(inname);
    if (image.empty()) {
        std::cerr << "Error: Could not read the image file." << std::endl;
        return 1;
    }
    Mat outimage;

    // 进行滤波
    if (mode == 'b' || mode == 'B') {
        BoxFilter(image, outimage, w, h);
    }
    else if (mode == 'g' || mode == 'G') {
        GaussianFilter(image, outimage, sigma);
    }
    else if (mode == 'm' || mode == 'M') {
        MedianFilter(image, outimage, w, h);
    }
    else if (mode == 'x' || mode == 'X') {
        BilateralFilter(image, outimage, sigma_s, sigma_r);
    }

    // 显示原始图像和处理后的图像
    imshow("Original", image);
    imshow("Filtered", outimage);
    // 保存处理后的图像
    imwrite(outname, outimage);
    // 关闭所有窗口
    waitKey(0);
    destroyAllWindows(); 
    return 0;
}
