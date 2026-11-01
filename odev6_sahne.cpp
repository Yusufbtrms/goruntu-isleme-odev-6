// Ödev 6 - Test sahnesi üretici: doğruları ve çemberleri KESİN bilinen bir resim üretir.
// Hough algoritmalarının ne kadar doğru çalıştığını gerçek değerlerle karşılaştırabilmek için kullanılır.
// Kendi fotoğrafınızı da kullanabilirsiniz; bu program sadece elinizde uygun resim yokken içindir.
//
// Sahne (800x600): 4 doğru, 2 tek çember ve 3 eş merkezli çember (merkez 560,320; yarıçaplar 40, 80, 120),
// açık zemin üstüne koyu çizgiler, hafif Gauss gürültüsü (sigma = 8).
//
// Kullanım: ./sahne [cikti_dosyasi=sahne.png]
// Derleme:  g++ -O2 odev6_sahne.cpp -o sahne.exe -std=c++17 $(pkg-config --cflags --libs opencv5)

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
    const std::string outPath = (argc >= 2) ? argv[1] : "sahne.png";
    const int W = 800, H = 600;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(210, 210, 210));
    const cv::Scalar ink(40, 40, 40);
    const int thickness = 3;

    const std::vector<std::pair<cv::Point, cv::Point>> lines = {
        {{40, 60}, {760, 60}}, {{740, 60}, {740, 540}}, {{40, 540}, {360, 220}}, {{420, 540}, {740, 380}}};
    struct Circle {
        cv::Point c;
        int r;
    };
    const std::vector<Circle> circles = {{{200, 200}, 70}, {{330, 430}, 60}, {{560, 320}, 40}, {{560, 320}, 80}, {{560, 320}, 120}};

    for (const auto& l : lines) cv::line(img, l.first, l.second, ink, thickness, cv::LINE_AA);
    for (const auto& c : circles) cv::circle(img, c.c, c.r, ink, thickness, cv::LINE_AA);

    // Hafif Gauss gürültüsü (sabit tohum: her çalıştırmada aynı resim)
    std::mt19937 rng(11);
    std::normal_distribution<double> noise(0.0, 8.0);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const double n = noise(rng);
            cv::Vec3b& p = img.at<cv::Vec3b>(y, x);
            for (int ch = 0; ch < 3; ++ch) p[ch] = cv::saturate_cast<uchar>(p[ch] + n);
        }
    }
    cv::imwrite(outPath, img);

    // Gerçek değerler: doğrular (rho, theta) ve çemberler (merkez, yarıçap)
    const double pi = 3.14159265358979323846;
    std::cout << "Sahne kaydedildi: " << outPath << " (" << W << "x" << H << ")\n\n";
    std::cout << "GERCEK DOGRULAR (x*cos(theta) + y*sin(theta) = rho, theta 0..180 derece):\n";
    std::cout << std::fixed << std::setprecision(1);
    for (const auto& l : lines) {
        const double dx = l.second.x - l.first.x, dy = l.second.y - l.first.y;
        const double len = std::sqrt(dx * dx + dy * dy);
        const double nx = -dy / len, ny = dx / len;
        double rho = nx * l.first.x + ny * l.first.y;
        double theta = std::atan2(ny, nx);
        if (theta < 0.0) {
            theta += pi;
            rho = -rho;
        } else if (theta >= pi) {
            theta -= pi;
            rho = -rho;
        }
        std::cout << "  rho = " << std::setw(6) << rho << ", theta = " << std::setw(5) << theta * 180.0 / pi << " derece\n";
    }
    std::cout << "\nGERCEK CEMBERLER (merkez x, merkez y, yaricap):\n";
    for (const auto& c : circles) std::cout << "  (" << c.c.x << ", " << c.c.y << ", " << c.r << ")\n";
    std::cout << "\nNot: son uc cember esmerkezlidir.\n";
    return 0;
}
