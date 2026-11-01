// Ödev 6 - Hough dönüşümleri: tüm programların ortak kullandığı yardımcı kodlar (sadece başlık dosyası).
// Bu dosyayı programlarla AYNI klasöre koyun; ayrıca derlemenize gerek yoktur.
//
// İçindekiler:
//   - makeEdges(): resim -> gri -> Gauss -> Canny -> median ile "kenar resmi"
//   - drawLineRhoTheta(): (rho, theta) ile verilen sonsuz doğruyu resme çizer
//   - toBgr(): gri resmi renkli çizim için 3 kanala çevirir

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace hough {

struct Edges {
    cv::Mat gray;     // gri seviye resim
    cv::Mat canny;    // Canny sonucu (0/255, 1 piksel kalınlığında kenarlar)
    cv::Mat medDirect;  // Canny sonucunun DOĞRUDAN median filtreden geçmiş hali (karşılaştırma için)
    cv::Mat cleaned;  // "kenar resmi": Canny -> 3x3 genişletme -> median (Hough'a verilen resim)
    int cannyCount = 0, medDirectCount = 0, cleanedCount = 0;
};

// Kenar resmini üretir.
// Hocanın istediği sıra: Canny -> median filtre. Ancak 1 piksel kalınlığındaki kenar çizgileri 3x3
// medyan filtrede silinir (bir çizgi pikselinin 3x3 komşuluğunda 9 pikselin sadece 3'ü doludur,
// medyan 0 çıkar). Bu yüzden Canny kenarları önce 3x3 genişletilip (dilate) sonra median uygulanır;
// böylece median kenarları silmez, sadece pürüzleri düzeltir. Doğrudan median sonucu karşılaştırma için
// medDirect olarak tutulur.
inline Edges makeEdges(const cv::Mat& img, double cannyLow = 60.0, double cannyHigh = 150.0, int medianSize = 3) {
    Edges e;
    if (img.channels() == 3) cv::cvtColor(img, e.gray, cv::COLOR_BGR2GRAY);
    else e.gray = img.clone();

    cv::Mat blur;
    cv::GaussianBlur(e.gray, blur, cv::Size(5, 5), 1.4);
    cv::Canny(blur, e.canny, cannyLow, cannyHigh);

    cv::medianBlur(e.canny, e.medDirect, medianSize);

    cv::Mat thick;
    cv::dilate(e.canny, thick, cv::Mat::ones(3, 3, CV_8UC1));
    cv::medianBlur(thick, e.cleaned, medianSize);

    e.cannyCount = cv::countNonZero(e.canny);
    e.medDirectCount = cv::countNonZero(e.medDirect);
    e.cleanedCount = cv::countNonZero(e.cleaned);
    return e;
}

// (rho, theta) ile verilen doğruyu çizer: x*cos(theta) + y*sin(theta) = rho
inline void drawLineRhoTheta(cv::Mat& img, double rho, double theta, const cv::Scalar& color, int thickness = 2) {
    const double a = std::cos(theta), b = std::sin(theta);
    const double x0 = a * rho, y0 = b * rho;
    const double L = img.cols + img.rows;
    const cv::Point p1(cvRound(x0 - L * b), cvRound(y0 + L * a));
    const cv::Point p2(cvRound(x0 + L * b), cvRound(y0 - L * a));
    cv::line(img, p1, p2, color, thickness, cv::LINE_AA);
}

inline cv::Mat toBgr(const cv::Mat& gray) {
    cv::Mat c;
    cv::cvtColor(gray, c, cv::COLOR_GRAY2BGR);
    return c;
}

}  // namespace hough
