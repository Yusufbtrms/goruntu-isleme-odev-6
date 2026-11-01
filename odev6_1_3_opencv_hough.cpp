// Ödev 6 - Madde 1 ve 3: OpenCV'nin hazır Hough fonksiyonlarıyla doğru ve çember bulma.
//
// Kenar resmi: Gauss -> Canny -> (3x3 genişletme) -> median filtre (ayrıntı için hough_ortak.hpp)
//
// Madde 1: İki farklı OpenCV Hough doğru fonksiyonu
//   - cv::HoughLines  : standart Hough, (rho, theta) ile sonsuz doğrular verir  -> KIRMIZI çizilir
//   - cv::HoughLinesP : olasılıksal Hough, uç noktalı doğru parçaları verir     -> YEŞİL çizilir
// Madde 3: cv::HoughCircles (HOUGH_GRADIENT)
//   - kenar resmi üzerinde  -> MACENTA çizilir
//   - karşılaştırma için gri resim üzerinde -> TURUNCU çizilir
//   Not: cv::HoughCircles kendi içinde Canny çalıştırır (param1 = Canny üst eşiği). Kenar resmini girdi
//   olarak verince "kenarların kenarı" bulunur; bu yüzden gri resimle de denenir.
//   Ayrıca merkezleri minDist'ten yakın çemberlerden sadece birini tutar: eş merkezli çemberler elenir.
//
// Kullanım: ./opencv_hough <goruntu> [cikti_dizini] [satir_esik=280] [cember_param2=30]
// Derleme:  g++ -O2 odev6_1_3_opencv_hough.cpp -o opencv_hough.exe -std=c++17 $(pkg-config --cflags --libs opencv5)
// (hough_ortak.hpp bu dosyayla aynı klasörde olmalıdır)

#include "hough_ortak.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Kullanim: " << argv[0] << " <goruntu> [cikti_dizini] [satir_esik=280] [cember_param2=30]\n";
        return 1;
    }
    const fs::path outDir = (argc >= 3) ? fs::path(argv[2]) : fs::path("cikti_opencv_hough");
    const int lineThreshold = (argc >= 4) ? std::stoi(argv[3]) : 280;
    const double circleParam2 = (argc >= 5) ? std::stod(argv[4]) : 30.0;
    fs::create_directories(outDir);

    const cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Hata: goruntu okunamadi: " << argv[1] << "\n";
        return 1;
    }

    const hough::Edges e = hough::makeEdges(img);
    std::cout << "Goruntu boyutu: " << img.cols << "x" << img.rows << "\n\n";
    std::cout << "KENAR RESMI (kenar piksel sayilari):\n";
    std::cout << "  Canny                                 : " << e.cannyCount << "\n";
    std::cout << "  Canny -> dogrudan median 3x3          : " << e.medDirectCount
              << "  (ince kenarlar medyanda silinir)\n";
    std::cout << "  Canny -> genisletme -> median 3x3     : " << e.cleanedCount << "  (Hough'a verilen kenar resmi)\n\n";

    const cv::Scalar red(0, 0, 255), green(0, 200, 0), magenta(255, 0, 255), orange(0, 140, 255);

    // ---- Madde 1: HoughLines (standart) ----
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(e.cleaned, lines, 1.0, CV_PI / 180.0, lineThreshold);
    cv::Mat outLines = img.clone();
    std::cout << "MADDE 1 - cv::HoughLines (esik = " << lineThreshold << "): " << lines.size() << " dogru\n";
    std::cout << std::fixed << std::setprecision(1);
    for (const auto& l : lines) {
        hough::drawLineRhoTheta(outLines, l[0], l[1], red, 2);
        std::cout << "  rho = " << std::setw(7) << l[0] << ", theta = " << std::setw(6) << l[1] * 180.0 / CV_PI << " derece\n";
    }

    // ---- Madde 1: HoughLinesP (olasılıksal) ----
    std::vector<cv::Vec4i> segs;
    cv::HoughLinesP(e.cleaned, segs, 1.0, CV_PI / 180.0, 80, 80.0, 15.0);
    cv::Mat outSegs = img.clone();
    for (const auto& s : segs) cv::line(outSegs, cv::Point(s[0], s[1]), cv::Point(s[2], s[3]), green, 2, cv::LINE_AA);
    std::cout << "\nMADDE 1 - cv::HoughLinesP (esik = 80, min uzunluk = 80, bosluk = 15): " << segs.size() << " dogru parcasi\n";

    // ---- Madde 3: HoughCircles ----
    std::vector<cv::Vec3f> circlesEdge, circlesGray;
    cv::HoughCircles(e.cleaned, circlesEdge, cv::HOUGH_GRADIENT, 1.0, 30.0, 100.0, circleParam2, 15, 150);
    cv::Mat grayBlur;
    cv::GaussianBlur(e.gray, grayBlur, cv::Size(9, 9), 2.0);
    cv::HoughCircles(grayBlur, circlesGray, cv::HOUGH_GRADIENT, 1.0, 30.0, 100.0, circleParam2, 15, 150);

    cv::Mat outCircEdge = img.clone(), outCircGray = img.clone();
    std::cout << "\nMADDE 3 - cv::HoughCircles (param1 = 100, param2 = " << circleParam2 << ", minDist = 30, yaricap 15..150)\n";
    std::cout << "  Kenar resmi uzerinde: " << circlesEdge.size() << " cember\n";
    for (const auto& c : circlesEdge) {
        cv::circle(outCircEdge, cv::Point(cvRound(c[0]), cvRound(c[1])), cvRound(c[2]), magenta, 2, cv::LINE_AA);
        cv::circle(outCircEdge, cv::Point(cvRound(c[0]), cvRound(c[1])), 3, magenta, -1, cv::LINE_AA);
        std::cout << "    merkez = (" << std::setw(6) << c[0] << ", " << std::setw(6) << c[1] << "), yaricap = " << std::setw(6) << c[2] << "\n";
    }
    std::cout << "  Gri resim uzerinde  : " << circlesGray.size() << " cember\n";
    for (const auto& c : circlesGray) {
        cv::circle(outCircGray, cv::Point(cvRound(c[0]), cvRound(c[1])), cvRound(c[2]), orange, 2, cv::LINE_AA);
        cv::circle(outCircGray, cv::Point(cvRound(c[0]), cvRound(c[1])), 3, orange, -1, cv::LINE_AA);
        std::cout << "    merkez = (" << std::setw(6) << c[0] << ", " << std::setw(6) << c[1] << "), yaricap = " << std::setw(6) << c[2] << "\n";
    }

    cv::imwrite((outDir / "1_canny.png").string(), e.canny);
    cv::imwrite((outDir / "2_canny_dogrudan_median.png").string(), e.medDirect);
    cv::imwrite((outDir / "3_kenar_resmi_genislet_median.png").string(), e.cleaned);
    cv::imwrite((outDir / "4_houghlines_kirmizi.png").string(), outLines);
    cv::imwrite((outDir / "5_houghlinesp_yesil.png").string(), outSegs);
    cv::imwrite((outDir / "6_houghcircles_kenar_resmi_macenta.png").string(), outCircEdge);
    cv::imwrite((outDir / "7_houghcircles_gri_turuncu.png").string(), outCircGray);

    std::cout << "\nKaydedilenler (" << outDir.string() << "):\n"
              << "  1_canny.png                              : Canny kenarlari\n"
              << "  2_canny_dogrudan_median.png              : Canny'nin dogrudan median'dan gecmis hali (ince kenarlar silinir)\n"
              << "  3_kenar_resmi_genislet_median.png        : Hough'a verilen kenar resmi\n"
              << "  4_houghlines_kirmizi.png                 : Madde 1, HoughLines (kirmizi)\n"
              << "  5_houghlinesp_yesil.png                  : Madde 1, HoughLinesP (yesil)\n"
              << "  6_houghcircles_kenar_resmi_macenta.png   : Madde 3, kenar resmi uzerinde HoughCircles (macenta)\n"
              << "  7_houghcircles_gri_turuncu.png           : Madde 3, gri resim uzerinde HoughCircles (turuncu)\n";
    return 0;
}
