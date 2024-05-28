#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

class CylindricalPanorama {
public:
    virtual bool makePanorama(vector<Mat>& img_vec, Mat& img_out, double f) = 0;
};

class Panorama3790 : public CylindricalPanorama {
public:
    bool makePanorama(vector<Mat>& img_vec, Mat& img_out, double f) {
        int n = img_vec.size();
        if (n < 2) {
            return false;
        }

        vector<Mat> cyl_vec(n);
        for (int i = 0; i < n; i++) {
            projectToCylinder(img_vec[i], cyl_vec[i], f);
            int width = cyl_vec[i].cols, height = cyl_vec[i].rows;
            copyMakeBorder(cyl_vec[i], cyl_vec[i], height / 2, height / 2, width * 2.5, width * 2.5, BORDER_CONSTANT, Scalar(0, 0, 0));
        }

        img_out = cyl_vec[n / 2];
        for (int i = n / 2 - 1; i >= 0; i--) {
            if (!merge(img_out, cyl_vec[i], img_out)) {
                return false;
            }
        }
        for (int i = n / 2 + 1; i < n; i++) {
            if (!merge(img_out, cyl_vec[i], img_out)) {
                return false;
            }
        }

        img_out = img_out(getValidRange(img_out));

        return true;
    }


private:
    Vec3b bilinearInterpolation(Mat const& img, double x, double y) {
        int x1 = floor(x), x2 = ceil(x);
        int y1 = floor(y), y2 = ceil(y);
        double dx = x - x1, dy = y - y1;

        Vec3b val = (1 - dx) * (1 - dy) * img.at<Vec3b>(y1, x1)
            + dx * (1 - dy) * img.at<Vec3b>(y1, x2)
            + (1 - dx) * dy * img.at<Vec3b>(y2, x1)
            + dx * dy * img.at<Vec3b>(y2, x2);

        return val;
    }

    void projectToCylinder(Mat const& img, Mat& img_cyl, double f) {
        int width = img.cols, height = img.rows;
        double r = f;
        double cyl_width = r * atan(width / 2.0 / f), cyl_height = r * height / 2.0 / f;
        int cyl_cols = ceil(2 * cyl_width), cyl_rows = ceil(2 * cyl_height);

        img_cyl = Mat::zeros(cyl_rows, cyl_cols, CV_8UC3);

        for (int i = 0; i < cyl_rows; i++) {
            double y = (i - cyl_height) / r;
            for (int j = 0; j < cyl_cols; j++) {
                double x = f * tan((j - cyl_width) / r);
                double src_x = x + width / 2.0;
                double src_y = y * sqrt(f * f + x * x) + height / 2.0;
                if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
                    img_cyl.at<Vec3b>(i, j) = bilinearInterpolation(img, src_x, src_y);
                }
            }
        }
    }

    Rect getValidRange(Mat const& img) {
        Mat mask;
        inRange(img, Scalar(1, 1, 1), Scalar(255, 255, 255), mask);
        vector<Point> pts;
        findNonZero(mask, pts);
        return boundingRect(pts);
    }

    bool merge(Mat const& img1, Mat const& img2, Mat& img_out) {
        auto orb = ORB::create();
        vector<KeyPoint> keypoints1, keypoints2;
        Mat descriptors1, descriptors2;
        orb->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
        orb->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

        auto bf = BFMatcher::create();
        vector<vector<DMatch>> matches;
        bf->knnMatch(descriptors1, descriptors2, matches, 2);

        vector<DMatch> good_matches;
        for (const auto& match : matches) {
            if (match[0].distance < 0.5 * match[1].distance) {
                good_matches.push_back(match[0]);
            }
        }

        if (good_matches.size() < 4) {
            return false;
        }

        vector<Point2f> pts1, pts2;
        for (const auto& match : good_matches) {
            pts1.push_back(keypoints1[match.queryIdx].pt);
            pts2.push_back(keypoints2[match.trainIdx].pt);
        }

        Mat H = findHomography(pts2, pts1, RANSAC);
        if (H.empty()) {
            return false;
        }

        Mat img2_warp;
        warpPerspective(img2, img2_warp, H, img1.size());

        Mat mask = Mat::zeros(img2_warp.size(), CV_8U);
        for (int i = 0; i < img2_warp.rows; i++) {
            for (int j = 0; j < img2_warp.cols; j++) {
                if (img2_warp.at<Vec3b>(i, j) != Vec3b(0, 0, 0)) {
                    mask.at<uchar>(i, j) = 255;
                }
            }
        }

        Mat img_out1, img_out2;
        img1.copyTo(img_out1);
        img2_warp.copyTo(img_out2, mask);

        Mat mask_out;
        bitwise_not(mask, mask_out);
        img_out1.copyTo(img_out, mask_out);
        img_out2.copyTo(img_out, mask);

        Rect range1 = getValidRange(img_out1);
        Rect range2 = getValidRange(img_out2);
        if (range1.x > range2.x) {
            swap(range1, range2);
        }

        double alpha = 1.0 / (range1.x + range1.width - range2.x);
        for (int i = range2.y; i < range2.y + range2.height; i++) {
            for (int j = range2.x; j < range1.x + range1.width; j++) {
                if (mask.at<uchar>(i, j) > 0 && mask_out.at<uchar>(i, j) > 0) {
                    double weight = (j - range2.x) * alpha;
                    img_out.at<Vec3b>(i, j) = (1 - weight) * img_out1.at<Vec3b>(i, j) + weight * img_out2.at<Vec3b>(i, j);
                }
            }
        }
        return true;
    }
};

int main(int argc, char* argv[]) {
    vector<Mat> img_vec;
    Panorama3790 pano;
    double f = 512.89;

    vector<String> filenames;
    glob("./image/data1/*.jpg", filenames); //data1 or data2
    if (filenames.size() < 2) {
        cerr << "Error: not enough images to stitch." << endl;
        return 1;
    }

    for (const auto& filename : filenames) {
        Mat img = imread(filename);
        if (img.empty()) {
            cerr << "Error: " << filename << " is not an image." << endl;
            return 1;
        }
        img_vec.push_back(img);
    }

    Mat img_out;
    if (!pano.makePanorama(img_vec, img_out, f)) {
        cerr << "Error." << endl;
        return 1;
    }
    cerr << "Success and saved." << endl;
    return 0;
}
