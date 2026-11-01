// Ödev 6 - Madde 2: Hough doğru dönüşümü, hazır fonksiyon kullanmadan, elle yazılmış.
//
// Doğru denklemi (normal biçimi):  rho = x * cos(theta) + y * sin(theta)
// Her kenar pikseli (x, y), kendisinden geçebilecek BÜTÜN doğrulara (her theta için bir rho) oy verir.
// Aynı doğru üzerindeki pikseller aynı (rho, theta) hücresine oy verir; o hücrenin oyu yükselir (tepe).
//
// Adımlar:
//   1) sin ve cos değerleri BİR KERE hesaplanıp bellekte tutulur (theta = 0..179 derece, 1 derece adım).
//   2) Oylama matrisi: satırlar rho (-D..+D, 1 piksel adım), sütunlar theta. D = resim köşegen uzunluğu.
//   3) Oylama: her kenar pikseli için tüm theta'larda rho hesaplanır, ilgili hücre 1 artırılır.
//      Oylama açıkça gösterilir (ilk piksellerin hangi hücrelere oy verdiği yazılır, oylamanın ilerleyişi
//      %1 / %10 / %100 anlarında resim olarak kaydedilir, aynı doğru üzerindeki 3 pikselin eğrileri
//      kesişim noktasıyla birlikte çizilir).
//   4) Eşik: oy sayısı eşik oranı x (en büyük oy) değerini aşan ve çevresindeki pencerede en büyük olan
//      hücreler (yerel maksimum) doğru sayılır. Tepe konumu çevresindeki oyların ağırlık merkeziyle inceltilir.
//   5) Bulunan doğrular orijinal resim üzerine çizilir.
//
// Kullanım: ./hough_dogru <goruntu> [cikti_dizini] [esik_orani=0.4]
// Derleme:  g++ -O2 odev6_2_hough_dogru.cpp -o hough_dogru.exe -std=c++17 $(pkg-config --cflags --libs opencv5)
// (hough_ortak.hpp bu dosyayla aynı klasörde olmalıdır)

#include "hough_ortak.hpp"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static const double PI = 3.14159265358979323846;

struct Peak {
    double rho;    // piksel
    double theta;  // derece (merkezleme sonrası kesirli, aralık dışı olabilir)
    int votes;
};

// Oylama matrisini (satır = rho, sütun = theta) resme çevirir; theta ekseni scaleX kat genişletilir.
static cv::Mat accToImage(const std::vector<int>& acc, int numRho, int numTheta, int scaleX, double maxVal, bool sqrtScale) {
    cv::Mat img(numRho, numTheta * scaleX, CV_8UC1);
    for (int r = 0; r < numRho; ++r) {
        uchar* row = img.ptr<uchar>(r);
        for (int t = 0; t < numTheta; ++t) {
            double v = maxVal > 0 ? acc[static_cast<size_t>(r) * numTheta + t] / maxVal : 0.0;
            if (sqrtScale) v = std::sqrt(v);
            const uchar g = cv::saturate_cast<uchar>(v * 255.0);
            for (int k = 0; k < scaleX; ++k) row[t * scaleX + k] = g;
        }
    }
    return img;
}

// (rho, theta) -> theta 0..180 aralığına, rho işareti buna göre
static void canonical(double& rho, double& thetaDeg) {
    while (thetaDeg < 0.0) {
        thetaDeg += 180.0;
        rho = -rho;
    }
    while (thetaDeg >= 180.0) {
        thetaDeg -= 180.0;
        rho = -rho;
    }
}

// İki doğrunun orijine en yakın noktaları arasındaki uzaklık (piksel)
static double footDistance(double r1, double t1, double r2, double t2) {
    const double a1 = t1 * PI / 180.0, a2 = t2 * PI / 180.0;
    return std::hypot(r1 * std::cos(a1) - r2 * std::cos(a2), r1 * std::sin(a1) - r2 * std::sin(a2));
}

static double angleDiff(double t1, double t2) {
    double d = std::fmod(std::fabs(t1 - t2), 180.0);
    return std::min(d, 180.0 - d);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Kullanim: " << argv[0] << " <goruntu> [cikti_dizini] [esik_orani=0.4]\n";
        return 1;
    }
    const std::string inPath = argv[1];
    const fs::path outDir = (argc >= 3) ? fs::path(argv[2]) : fs::path("cikti_hough_dogru");
    const double thresholdRatio = (argc >= 4) ? std::stod(argv[3]) : 0.4;
    fs::create_directories(outDir);

    const cv::Mat img = cv::imread(inPath, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Hata: goruntu okunamadi: " << inPath << "\n";
        return 1;
    }
    const hough::Edges e = hough::makeEdges(img);
    const cv::Mat& edge = e.cleaned;
    const int W = img.cols, H = img.rows;

    // ---- Oylama matrisinin boyutları ----
    const int D = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(W) * W + static_cast<double>(H) * H)));
    const int numRho = 2 * D + 1;  // rho = -D .. +D
    const int numTheta = 180;      // theta = 0 .. 179 derece
    std::cout << "Goruntu boyutu: " << W << "x" << H << ", kenar piksel sayisi: " << e.cleanedCount << "\n";
    std::cout << "Oylama matrisi: " << numRho << " (rho: -" << D << ".." << D << ") x " << numTheta << " (theta: 0..179 derece), "
              << static_cast<double>(numRho) * numTheta * sizeof(int) / 1e6 << " MB\n\n";

    // ---- 1) sin ve cos tablosu: bir kere hesaplanır, bellekte tutulur ----
    std::vector<double> cosT(numTheta), sinT(numTheta);
    for (int t = 0; t < numTheta; ++t) {
        const double a = t * PI / 180.0;
        cosT[t] = std::cos(a);
        sinT[t] = std::sin(a);
    }
    std::cout << "sin/cos tablosu (" << numTheta << " giris) bir kere hesaplandi. Ornek:\n";
    std::cout << std::fixed << std::setprecision(4);
    for (int t : {0, 45, 90, 135}) std::cout << "  theta = " << std::setw(3) << t << " derece: cos = " << std::setw(7) << cosT[t] << ", sin = " << std::setw(7) << sinT[t] << "\n";

    // ---- 2) kenar piksellerini listele (oylama sırası karışık: ilerleyişin görülebilmesi için) ----
    std::vector<cv::Point> pts;
    pts.reserve(static_cast<size_t>(e.cleanedCount));
    for (int y = 0; y < H; ++y) {
        const uchar* row = edge.ptr<uchar>(y);
        for (int x = 0; x < W; ++x)
            if (row[x]) pts.emplace_back(x, y);
    }
    std::mt19937 rng(5);
    std::shuffle(pts.begin(), pts.end(), rng);

    // ---- 3) OYLAMA: bu adım açıkça gösterilir ----
    std::cout << "\nOYLAMA ORNEGI: ilk iki kenar pikselinin her theta icin verdigi oy (rho = x*cos(theta) + y*sin(theta)):\n";
    for (int i = 0; i < 2 && i < static_cast<int>(pts.size()); ++i) {
        std::cout << "  Kenar pikseli (x = " << pts[i].x << ", y = " << pts[i].y << "):\n";
        for (int t : {0, 30, 60, 90, 120, 150}) {
            const double rho = pts[i].x * cosT[t] + pts[i].y * sinT[t];
            std::cout << "    theta = " << std::setw(3) << t << " derece -> rho = " << std::setw(8) << rho << " -> hucre (rho = "
                      << std::setw(5) << std::lround(rho) << ", theta = " << std::setw(3) << t << ") +1 oy\n";
        }
    }

    std::vector<int> acc(static_cast<size_t>(numRho) * numTheta, 0);
    std::vector<int> accSnap1, accSnap10;
    const size_t snap1 = std::max<size_t>(1, pts.size() / 100), snap10 = std::max<size_t>(1, pts.size() / 10);

    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < pts.size(); ++i) {
        const int x = pts[i].x, y = pts[i].y;
        for (int t = 0; t < numTheta; ++t) {
            const double rho = x * cosT[t] + y * sinT[t];            // bu piksel icin bu theta'daki rho
            const int r = static_cast<int>(std::lround(rho)) + D;    // matris satiri
            ++acc[static_cast<size_t>(r) * numTheta + t];            // OY VER
        }
        if (i + 1 == snap1) accSnap1 = acc;
        if (i + 1 == snap10) accSnap10 = acc;
    }
    const auto t1 = std::chrono::steady_clock::now();

    int maxVotes = 0;
    long long nonZero = 0;
    for (int v : acc) {
        maxVotes = std::max(maxVotes, v);
        nonZero += (v > 0) ? 1 : 0;
    }
    std::cout << std::setprecision(1);
    std::cout << "\nOylama tamamlandi: " << pts.size() << " piksel x " << numTheta << " theta = " << pts.size() * numTheta
              << " oy, sure = " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
    std::cout << "Oy alan hucre sayisi: " << nonZero << " / " << acc.size() << ", en yuksek oy: " << maxVotes << "\n";

    // ---- 4) Eşik ve yerel maksimum ----
    const int threshold = std::max(1, static_cast<int>(std::lround(thresholdRatio * maxVotes)));
    const int lowVotes = std::max(1, maxVotes / 10);
    const int halfRho = 10, halfTheta = 6;  // tepe seçim penceresi

    // theta 0 / 180 sınırında (rho, theta) = (-rho, theta +/- 180) olduğundan sarmalama yapılır
    auto at = [&](int r, int t) -> int {
        if (t < 0) {
            t += numTheta;
            r = 2 * D - r;
        } else if (t >= numTheta) {
            t -= numTheta;
            r = 2 * D - r;
        }
        if (r < 0 || r >= numRho) return 0;
        return acc[static_cast<size_t>(r) * numTheta + t];
    };

    std::vector<Peak> peaks;
    for (int r = 0; r < numRho; ++r) {
        for (int t = 0; t < numTheta; ++t) {
            const int v = acc[static_cast<size_t>(r) * numTheta + t];
            if (v < lowVotes) continue;
            bool isPeak = true;
            for (int dr = -halfRho; dr <= halfRho && isPeak; ++dr) {
                for (int dt = -halfTheta; dt <= halfTheta; ++dt) {
                    if (dr == 0 && dt == 0) continue;
                    const int nv = at(r + dr, t + dt);
                    // plato durumunda tek hücre seçilsin: eşit komşulardan sadece sıralamada ilk gelen kalır
                    if (nv > v || (nv == v && (dr < 0 || (dr == 0 && dt < 0)))) {
                        isPeak = false;
                        break;
                    }
                }
            }
            if (!isPeak) continue;

            // Ağırlık merkeziyle inceltme (tepenin yarısından güçlü komşu oylar)
            double sw = 0, sr = 0, st = 0;
            for (int dr = -4; dr <= 4; ++dr) {
                for (int dt = -3; dt <= 3; ++dt) {
                    const int w = at(r + dr, t + dt);
                    if (w * 2 >= v) {
                        sw += w;
                        sr += static_cast<double>(w) * (r + dr - D);
                        st += static_cast<double>(w) * (t + dt);
                    }
                }
            }
            peaks.push_back({sr / sw, st / sw, v});
        }
    }
    std::sort(peaks.begin(), peaks.end(), [](const Peak& a, const Peak& b) { return a.votes > b.votes; });

    std::cout << "\nESIK: " << thresholdRatio << " x " << maxVotes << " = " << threshold << " oy\n";
    std::cout << "En guclu yerel tepeler (ilk 12; * = esigi gecen, yani doguru olarak secilen):\n";
    int shown = 0, detected = 0;
    std::vector<Peak> lines;
    for (const Peak& p : peaks) {
        if (p.votes >= threshold) {
            lines.push_back(p);
            ++detected;
        }
        if (shown < 12) {
            double rho = p.rho, th = p.theta;
            canonical(rho, th);
            std::cout << "  " << (p.votes >= threshold ? "* " : "  ") << "rho = " << std::setw(7) << rho << ", theta = " << std::setw(6) << th
                      << " derece, oy = " << p.votes << "\n";
            ++shown;
        }
    }
    std::cout << "Esigi gecen dogru sayisi: " << detected << "\n";

    // Gerçek değerlerle karşılaştırma (sahne.png için)
    if (inPath.find("sahne") != std::string::npos) {
        const double truth[4][2] = {{60.0, 90.0}, {740.0, 0.0}, {410.1, 45.0}, {670.8, 63.4}};
        std::cout << "\nGERCEK DOGRULARLA KARSILASTIRMA (sahne.png):\n";
        for (const auto& tr : truth) {
            int best = -1;
            double bestD = 1e18;
            for (size_t i = 0; i < lines.size(); ++i) {
                const double d = footDistance(tr[0], tr[1], lines[i].rho, lines[i].theta);
                if (d < bestD) {
                    bestD = d;
                    best = static_cast<int>(i);
                }
            }
            if (best < 0) {
                std::cout << "  gercek (rho = " << tr[0] << ", theta = " << tr[1] << "): BULUNAMADI\n";
                continue;
            }
            double rho = lines[best].rho, th = lines[best].theta;
            canonical(rho, th);
            std::cout << "  gercek (rho = " << std::setw(6) << tr[0] << ", theta = " << std::setw(5) << tr[1] << ") -> bulunan (rho = "
                      << std::setw(7) << rho << ", theta = " << std::setw(6) << th << "), aci farki = " << angleDiff(tr[1], lines[best].theta)
                      << " derece, orijine en yakin noktalar arasi = " << bestD << " piksel\n";
        }
    }

    // ---- 5) Görseller ----
    // (a) oylama matrisi
    const int scaleX = 4;
    const cv::Mat accImg = accToImage(acc, numRho, numTheta, scaleX, maxVotes, false);
    const cv::Mat accImgSqrt = accToImage(acc, numRho, numTheta, scaleX, maxVotes, true);
    cv::Mat accPeaks = hough::toBgr(accImgSqrt);
    for (const Peak& p : lines) {
        const cv::Point c(static_cast<int>(std::lround(p.theta * scaleX + scaleX / 2.0)), static_cast<int>(std::lround(p.rho + D)));
        cv::circle(accPeaks, c, 14, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    // (b) oylamanın ilerleyişi: %1, %10, %100 (karekök ölçekli)
    cv::Mat progress;
    cv::hconcat(std::vector<cv::Mat>{accToImage(accSnap1, numRho, numTheta, 2, maxVotes, true),
                                     accToImage(accSnap10, numRho, numTheta, 2, maxVotes, true),
                                     accToImage(acc, numRho, numTheta, 2, maxVotes, true)},
                progress);

    // (c) en güçlü doğru üzerindeki 3 pikselin oy eğrileri (sinüzoidler) ve kesişim noktası
    cv::Mat demo = cv::Mat::zeros(numRho, numTheta * scaleX, CV_8UC3);
    if (!lines.empty()) {
        const double pr = lines[0].rho, pt = lines[0].theta * PI / 180.0;
        std::vector<std::pair<double, cv::Point>> onLine;
        for (const cv::Point& p : pts) {
            if (std::fabs(p.x * std::cos(pt) + p.y * std::sin(pt) - pr) < 1.0)
                onLine.push_back({-p.x * std::sin(pt) + p.y * std::cos(pt), p});  // doğru boyunca konum
        }
        if (onLine.size() >= 3) {
            std::sort(onLine.begin(), onLine.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            const cv::Point sel[3] = {onLine.front().second, onLine[onLine.size() / 2].second, onLine.back().second};
            const cv::Scalar colors[3] = {cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 160, 0)};
            std::cout << "\nOY EGRILERI: en guclu dogru uzerinden 3 kenar pikseli secildi:\n";
            for (int k = 0; k < 3; ++k) {
                std::cout << "  piksel " << (k + 1) << ": (x = " << sel[k].x << ", y = " << sel[k].y << ")\n";
                cv::Point prev;
                for (int t = 0; t < numTheta; ++t) {
                    const double rho = sel[k].x * cosT[t] + sel[k].y * sinT[t];
                    const cv::Point cur(t * scaleX + scaleX / 2, static_cast<int>(std::lround(rho)) + D);
                    if (t > 0) cv::line(demo, prev, cur, colors[k], 2, cv::LINE_AA);
                    prev = cur;
                }
            }
            double rho = lines[0].rho, th = lines[0].theta;
            if (th >= 0.0 && th < numTheta)
                cv::circle(demo, cv::Point(static_cast<int>(std::lround(th * scaleX + scaleX / 2.0)), static_cast<int>(std::lround(rho + D))), 16,
                           cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
            canonical(rho, th);
            std::cout << "  uc egri beyaz daire icindeki tek noktada kesisiyor: rho = " << rho << ", theta = " << th << " derece\n";
        }
    }

    // (d) bulunan doğrular orijinal resim üzerinde (SARI)
    cv::Mat out = img.clone();
    for (const Peak& p : lines) hough::drawLineRhoTheta(out, p.rho, p.theta * PI / 180.0, cv::Scalar(0, 255, 255), 2);

    cv::imwrite((outDir / "0_kenar_resmi.png").string(), edge);
    cv::imwrite((outDir / "1_oylama_matrisi.png").string(), accImg);
    cv::imwrite((outDir / "2_oylama_matrisi_karekok_olcekli.png").string(), accImgSqrt);
    cv::imwrite((outDir / "3_oylama_matrisi_secilen_tepeler.png").string(), accPeaks);
    cv::imwrite((outDir / "4_oylama_ilerleyisi_yuzde1_10_100.png").string(), progress);
    cv::imwrite((outDir / "5_uc_pikselin_oy_egrileri.png").string(), demo);
    cv::imwrite((outDir / "6_bulunan_dogrular_sari.png").string(), out);

    std::cout << "\nKaydedilenler (" << outDir.string() << "):\n"
              << "  0_kenar_resmi.png                          : oylamada kullanilan kenar pikselleri\n"
              << "  1_oylama_matrisi.png                       : oylama matrisi (yatay: theta 0..179 derece, dikey: rho -" << D << ".." << D << ")\n"
              << "  2_oylama_matrisi_karekok_olcekli.png       : ayni matris, zayif yapilar gorunsun diye karekok olcekli\n"
              << "  3_oylama_matrisi_secilen_tepeler.png       : esigi gecen tepeler kirmizi daireyle isaretli\n"
              << "  4_oylama_ilerleyisi_yuzde1_10_100.png      : pikseller oy verdikce matrisin olusumu (soldan saga %1, %10, %100)\n"
              << "  5_uc_pikselin_oy_egrileri.png              : ayni dogru uzerindeki 3 pikselin sinuzoidleri tek noktada kesisir\n"
              << "  6_bulunan_dogrular_sari.png                : Madde 2 sonucu: bulunan dogrular orijinal resim uzerinde (sari)\n";
    return 0;
}
