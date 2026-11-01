// Ödev 6 - Madde 4: Hough çember dönüşümü, hazır fonksiyon kullanmadan, elle yazılmış.
//
// Çember denklemi:  (x - a)^2 + (y - b)^2 = r^2   ->   x = a + r*cos(theta),  y = b + r*sin(theta)
// Bir kenar pikseli (x, y) belirli bir r yarıçapı için, merkezi (a, b) = (x - r*cos(theta), y - r*sin(theta))
// olan bütün çemberlere oy verir; yani kendisinin etrafında r yarıçaplı bir çember üzerindeki bütün
// merkez adaylarına. Gerçek bir çemberin üzerindeki pikseller bu merkez çemberlerinin ortak noktası olan
// gerçek merkeze oy verir; orada oy yığılır.
//
// Adımlar:
//   1) sin ve cos değerleri BİR KERE hesaplanıp bellekte tutulur (1440 giriş, 0.25 derece adım).
//   2) 3 boyutlu oylama matrisi: (r, b, a) = (yarıçap, merkez y, merkez x).
//   3) Oylama: her yarıçap için, her kenar pikseli, o yarıçaptaki oy çemberi üzerindeki merkez adaylarına
//      oy verir. Oy çemberi, komşu oylar arasında yaklaşık 1 piksel yay aralığı kalacak kadar örneklenir
//      (r yarıçapında yaklaşık 2*pi*r örnek). Böylece her yarıçapta bir piksel bir hücreye benzer sayıda oy
//      verir; küçük yarıçaplar yapay olarak avantajlı olmaz.
//      Oylama açıkça gösterilir (ilk piksellerin oyları yazılır; 3 pikselin oy çemberleri tek noktada
//      kesişirken çizilir; oylama matrisinin yarıçap boyunca en büyük değer izdüşümü ve dilimi kaydedilir).
//   4) Oy sayıları 3x3 komşulukta toplanır (piksel yuvarlama farklarını toplamak için).
//      Skor = oy / (2*pi*r): "çemberin çevresinin ne kadarı kenar pikselleriyle destekleniyor" ölçüsü.
//      Eşik: skor, en yüksek skorun eşik oranı katını aşan ve 3 boyutlu pencerede en büyük olan hücreler
//      çember sayılır.
//   5) Bulunan çemberler orijinal resim üzerine çizilir.
//
// Oylama için kenar resmi, Canny -> genişletme -> median sonucunun 3x3 aşındırılmasıyla (erode)
// inceltilmiş halidir; kalın kenarlar oyları gereksiz yere yayar.
//
// Kullanım: ./hough_cember <goruntu> [cikti_dizini] [esik_orani=0.6] [r_min=30] [r_max=140]
// Derleme:  g++ -O2 odev6_4_hough_cember.cpp -o hough_cember.exe -std=c++17 $(pkg-config --cflags --libs opencv5)
// (hough_ortak.hpp bu dosyayla aynı klasörde olmalıdır)

#include "hough_ortak.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static const double PI = 3.14159265358979323846;

struct Peak {
    double a, b, r;
    int votes;
    double score;  // oy / (2*pi*r)
};

// 3x3 komşuluk toplamı (yerinde): oyları çevre hücrelerle birleştirir
static void boxFilter3x3(uint16_t* s, int W, int H, std::vector<int>& tmp) {
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int sum = s[static_cast<size_t>(y) * W + x];
            if (x > 0) sum += s[static_cast<size_t>(y) * W + x - 1];
            if (x + 1 < W) sum += s[static_cast<size_t>(y) * W + x + 1];
            tmp[static_cast<size_t>(y) * W + x] = sum;
        }
    }
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int sum = tmp[static_cast<size_t>(y) * W + x];
            if (y > 0) sum += tmp[static_cast<size_t>(y - 1) * W + x];
            if (y + 1 < H) sum += tmp[static_cast<size_t>(y + 1) * W + x];
            s[static_cast<size_t>(y) * W + x] = static_cast<uint16_t>(std::min(sum, 65535));
        }
    }
}

static cv::Mat sliceToImage(const uint16_t* s, int W, int H, double maxVal) {
    cv::Mat img(H, W, CV_8UC1);
    for (int y = 0; y < H; ++y) {
        uchar* row = img.ptr<uchar>(y);
        for (int x = 0; x < W; ++x) {
            const double v = maxVal > 0 ? std::sqrt(static_cast<double>(s[static_cast<size_t>(y) * W + x]) / maxVal) : 0.0;
            row[x] = cv::saturate_cast<uchar>(v * 255.0);
        }
    }
    return img;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Kullanim: " << argv[0] << " <goruntu> [cikti_dizini] [esik_orani=0.6] [r_min=30] [r_max=140]\n";
        return 1;
    }
    const std::string inPath = argv[1];
    const fs::path outDir = (argc >= 3) ? fs::path(argv[2]) : fs::path("cikti_hough_cember");
    const double thresholdRatio = (argc >= 4) ? std::stod(argv[3]) : 0.6;
    const int rMin = (argc >= 5) ? std::stoi(argv[4]) : 30;
    const int rMax = (argc >= 6) ? std::stoi(argv[5]) : 140;
    const int rStep = 1;
    fs::create_directories(outDir);

    const cv::Mat img = cv::imread(inPath, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Hata: goruntu okunamadi: " << inPath << "\n";
        return 1;
    }
    const int W = img.cols, H = img.rows;
    const hough::Edges e = hough::makeEdges(img);

    // Oylama için inceltilmiş kenar resmi
    cv::Mat thin;
    cv::erode(e.cleaned, thin, cv::Mat::ones(3, 3, CV_8UC1));
    std::vector<cv::Point> pts;
    for (int y = 0; y < H; ++y) {
        const uchar* row = thin.ptr<uchar>(y);
        for (int x = 0; x < W; ++x)
            if (row[x]) pts.emplace_back(x, y);
    }

    const int numTheta = 1440;  // sin/cos tablosunun çözünürlüğü: 0.25 derece
    const int nR = (rMax - rMin) / rStep + 1;
    std::cout << "Goruntu boyutu: " << W << "x" << H << "\n";
    std::cout << "Kenar resmi: kalin " << e.cleanedCount << " piksel -> inceltilmis " << pts.size() << " piksel (oylamada kullanilan)\n";
    std::cout << "Oylama matrisi (3 boyutlu): yaricap " << rMin << ".." << rMax << " (" << nR << " dilim) x " << H << " x " << W << " = "
              << static_cast<double>(nR) * W * H * sizeof(uint16_t) / 1e6 << " MB\n\n";

    // ---- 1) sin ve cos tablosu: bir kere hesaplanır, bellekte tutulur ----
    std::vector<double> cosT(numTheta), sinT(numTheta);
    for (int t = 0; t < numTheta; ++t) {
        const double a = t * 2.0 * PI / numTheta;
        cosT[t] = std::cos(a);
        sinT[t] = std::sin(a);
    }
    std::cout << "sin/cos tablosu (" << numTheta << " giris, 0.25 derece adim) bir kere hesaplandi.\n";

    // ---- 2) OYLAMA ORNEGI (acikca gosterim) ----
    std::cout << "\nOYLAMA ORNEGI: ilk iki kenar pikselinin r = 70 icin verdigi oylar (merkez adayi = (x - r*cos(theta), y - r*sin(theta))):\n";
    std::cout << std::fixed << std::setprecision(1);
    for (int i = 0; i < 2 && i < static_cast<int>(pts.size()); ++i) {
        std::cout << "  Kenar pikseli (x = " << pts[i].x << ", y = " << pts[i].y << "):\n";
        for (int deg : {0, 60, 120, 180, 240, 300}) {
            const int t = deg * numTheta / 360;
            const double a = pts[i].x - 70.0 * cosT[t], b = pts[i].y - 70.0 * sinT[t];
            std::cout << "    theta = " << std::setw(3) << deg << " derece -> merkez adayi (a = " << std::setw(7) << a << ", b = " << std::setw(7) << b
                      << ") -> hucre (a = " << std::lround(a) << ", b = " << std::lround(b) << ", r = 70) +1 oy\n";
        }
    }

    // ---- 3) OYLAMA ----
    std::vector<uint16_t> acc(static_cast<size_t>(nR) * W * H, 0);
    std::vector<int> dx, dy;
    double totalVotes = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int ri = 0; ri < nR; ++ri) {
        const int r = rMin + ri * rStep;
        const int n = std::max(8, static_cast<int>(std::lround(2.0 * PI * r)));  // bu yaricapta oy cemberi ornek sayisi
        dx.assign(n, 0);
        dy.assign(n, 0);
        for (int i = 0; i < n; ++i) {  // sin/cos tablosundan, bu yaricap icin tam sayi kaydirmalar
            const int t = static_cast<int>(static_cast<long long>(i) * numTheta / n);
            dx[i] = static_cast<int>(std::lround(r * cosT[t]));
            dy[i] = static_cast<int>(std::lround(r * sinT[t]));
        }
        uint16_t* slice = &acc[static_cast<size_t>(ri) * W * H];
        for (const cv::Point& p : pts) {
            for (int i = 0; i < n; ++i) {
                const int a = p.x - dx[i];  // merkez adayi x
                const int b = p.y - dy[i];  // merkez adayi y
                if (static_cast<unsigned>(a) < static_cast<unsigned>(W) && static_cast<unsigned>(b) < static_cast<unsigned>(H))
                    ++slice[static_cast<size_t>(b) * W + a];  // OY VER
            }
        }
        totalVotes += static_cast<double>(pts.size()) * n;
    }
    const auto t1 = std::chrono::steady_clock::now();
    std::cout << "\nOylama tamamlandi: " << pts.size() << " piksel x " << nR << " yaricap, toplam " << totalVotes << " oy hesaplandi, sure = "
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    // Yaricap boyunca en buyuk deger izdusumu (hangi (a, b) konumlari herhangi bir yaricapta guclu oy aldi)
    std::vector<uint16_t> mip(static_cast<size_t>(W) * H, 0);
    for (int ri = 0; ri < nR; ++ri) {
        const uint16_t* slice = &acc[static_cast<size_t>(ri) * W * H];
        for (size_t i = 0; i < static_cast<size_t>(W) * H; ++i) mip[i] = std::max(mip[i], slice[i]);
    }
    int mipMax = 0;
    for (uint16_t v : mip) mipMax = std::max<int>(mipMax, v);
    std::cout << "Ham oylamada en yuksek tek hucre oyu: " << mipMax << " (3x3 toplama oncesi)\n";

    // ---- 4) 3x3 komsuluk toplami, skor, esik ve 3 boyutlu yerel maksimum ----
    std::vector<int> tmp(static_cast<size_t>(W) * H);
    for (int ri = 0; ri < nR; ++ri) boxFilter3x3(&acc[static_cast<size_t>(ri) * W * H], W, H, tmp);

    // Skor = oy / (2*pi*r). Her yaricap diliminde en buyuk degerden en yuksek skoru bul
    double scoreMax = 0;
    for (int ri = 0; ri < nR; ++ri) {
        const uint16_t* slice = &acc[static_cast<size_t>(ri) * W * H];
        int mv = 0;
        for (size_t i = 0; i < static_cast<size_t>(W) * H; ++i) mv = std::max<int>(mv, slice[i]);
        scoreMax = std::max(scoreMax, mv / (2.0 * PI * (rMin + ri * rStep)));
    }
    const double scoreThreshold = thresholdRatio * scoreMax;
    std::vector<int> lowVotes(nR);
    for (int ri = 0; ri < nR; ++ri) lowVotes[ri] = std::max(1, static_cast<int>(0.25 * scoreMax * 2.0 * PI * (rMin + ri * rStep)));

    auto at = [&](int ri, int b, int a) -> int {
        if (ri < 0 || ri >= nR || b < 0 || b >= H || a < 0 || a >= W) return 0;
        return acc[(static_cast<size_t>(ri) * H + b) * W + a];
    };
    const int hr = 7, hxy = 6;  // pencere: +/-7 yaricap (ic ve dis kenar halkalarini birlestirir), +/-6 piksel

    auto isLocalMax = [&](int ri, int b, int a, int v) {
        for (int dri = -hr; dri <= hr; ++dri)
            for (int db = -hxy; db <= hxy; ++db)
                for (int da = -hxy; da <= hxy; ++da) {
                    if (dri == 0 && db == 0 && da == 0) continue;
                    const int nv = at(ri + dri, b + db, a + da);
                    if (nv > v) return false;
                    if (nv == v && (dri < 0 || (dri == 0 && (db < 0 || (db == 0 && da < 0))))) return false;
                }
        return true;
    };

    std::vector<Peak> peaks;
    for (int ri = 0; ri < nR; ++ri) {
        for (int b = 0; b < H; ++b) {
            for (int a = 0; a < W; ++a) {
                const int v = acc[(static_cast<size_t>(ri) * H + b) * W + a];
                if (v < lowVotes[ri] || !isLocalMax(ri, b, a, v)) continue;

                double sw = 0, sa = 0, sb = 0;  // merkez: komsu oylarin agirlik merkezi
                for (int db = -3; db <= 3; ++db)
                    for (int da = -3; da <= 3; ++da) {
                        const int w = at(ri, b + db, a + da);
                        if (w * 2 >= v) {
                            sw += w;
                            sa += static_cast<double>(w) * (a + da);
                            sb += static_cast<double>(w) * (b + db);
                        }
                    }
                double rw = 0, rs = 0;  // yaricap: komsu yaricaplardaki oylarin agirlik ortalamasi
                for (int dri = -hr; dri <= hr; ++dri) {
                    const int w = at(ri + dri, b, a);
                    if (w * 2 >= v) {
                        rw += w;
                        rs += static_cast<double>(w) * (rMin + (ri + dri) * rStep);
                    }
                }
                peaks.push_back({sa / sw, sb / sw, rs / rw, v, v / (2.0 * PI * (rMin + ri * rStep))});
            }
        }
    }
    std::sort(peaks.begin(), peaks.end(), [](const Peak& x, const Peak& y) { return x.score > y.score; });

    std::cout << "\nESIK: en yuksek skor = " << std::setprecision(2) << scoreMax << " (oy / (2*pi*r)), esik = " << thresholdRatio << " x " << scoreMax << " = "
              << scoreThreshold << "\n";
    std::cout << "En guclu yerel tepeler (ilk 14; * = esigi gecen, yani cember olarak secilen):\n";
    std::cout << std::setprecision(1);
    std::vector<Peak> circles;
    int shown = 0;
    for (const Peak& p : peaks) {
        if (p.score >= scoreThreshold) circles.push_back(p);
        if (shown < 14) {
            std::cout << "  " << (p.score >= scoreThreshold ? "* " : "  ") << "merkez = (" << std::setw(6) << p.a << ", " << std::setw(6) << p.b
                      << "), yaricap = " << std::setw(6) << p.r << ", skor = " << std::setprecision(2) << std::setw(5) << p.score << std::setprecision(1)
                      << ", oy = " << p.votes << "\n";
            ++shown;
        }
    }
    std::cout << "Esigi gecen cember sayisi: " << circles.size() << " (yerel tepe sayisi: " << peaks.size() << ")\n";

    // Gerçek değerlerle karşılaştırma (sahne.png için)
    if (inPath.find("sahne") != std::string::npos) {
        const double truth[5][3] = {{200, 200, 70}, {330, 430, 60}, {560, 320, 40}, {560, 320, 80}, {560, 320, 120}};
        std::cout << "\nGERCEK CEMBERLERLE KARSILASTIRMA (sahne.png):\n";
        for (const auto& tr : truth) {
            int best = -1;
            double bestScore = 1e18;
            for (size_t i = 0; i < circles.size(); ++i) {
                const double d = std::hypot(circles[i].a - tr[0], circles[i].b - tr[1]) + std::fabs(circles[i].r - tr[2]);
                if (d < bestScore) {
                    bestScore = d;
                    best = static_cast<int>(i);
                }
            }
            std::cout << "  gercek (" << std::setw(4) << tr[0] << ", " << std::setw(4) << tr[1] << ", r = " << std::setw(3) << tr[2] << ") -> ";
            if (best < 0) {
                std::cout << "BULUNAMADI\n";
            } else {
                std::cout << "bulunan (" << std::setw(6) << circles[best].a << ", " << std::setw(6) << circles[best].b << ", r = " << std::setw(6)
                          << circles[best].r << "), merkez hatasi = " << std::hypot(circles[best].a - tr[0], circles[best].b - tr[1])
                          << " piksel, yaricap hatasi = " << std::fabs(circles[best].r - tr[2]) << " piksel\n";
            }
        }
    }

    // ---- 5) Görseller ----
    cv::Mat mipImg = sliceToImage(mip.data(), W, H, mipMax);
    cv::Mat mipColor = hough::toBgr(mipImg);

    cv::Mat sliceImg;
    if (!circles.empty()) {
        const int ri = std::min(nR - 1, std::max(0, static_cast<int>(std::lround((circles[0].r - rMin) / rStep))));
        const uint16_t* sl = &acc[static_cast<size_t>(ri) * W * H];
        sliceImg = hough::toBgr(sliceToImage(sl, W, H, circles[0].votes));
        cv::circle(sliceImg, cv::Point(static_cast<int>(std::lround(circles[0].a)), static_cast<int>(std::lround(circles[0].b))), 12,
                   cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    // ayni cember uzerindeki 3 pikselin oy cemberleri tek noktada kesisir
    cv::Mat demo = hough::toBgr(thin);
    demo *= 0.35;
    if (!circles.empty()) {
        const Peak& c = circles[0];
        const cv::Scalar colors[3] = {cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 160, 0)};
        std::cout << "\nOY CEMBERLERI: en guclu cemberin (merkez = (" << c.a << ", " << c.b << "), r = " << c.r << ") uzerinden 3 kenar pikseli secildi:\n";
        for (int k = 0; k < 3; ++k) {
            const double phi = k * 2.0 * PI / 3.0;
            const double ex = c.a + c.r * std::cos(phi), ey = c.b + c.r * std::sin(phi);
            cv::Point best(0, 0);
            double bd = 1e18;
            for (const cv::Point& p : pts) {
                const double d = std::hypot(p.x - ex, p.y - ey);
                if (d < bd) {
                    bd = d;
                    best = p;
                }
            }
            std::cout << "  piksel " << (k + 1) << ": (x = " << best.x << ", y = " << best.y << ") -> bu pikselin oy cemberi: merkezi pikselin kendisi, yaricapi "
                      << std::lround(c.r) << "\n";
            cv::circle(demo, best, static_cast<int>(std::lround(c.r)), colors[k], 2, cv::LINE_AA);
            cv::circle(demo, best, 4, colors[k], -1, cv::LINE_AA);
        }
        cv::circle(demo, cv::Point(static_cast<int>(std::lround(c.a)), static_cast<int>(std::lround(c.b))), 10, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        std::cout << "  uc oy cemberi beyaz daire icindeki tek noktada kesisiyor: merkez = (" << c.a << ", " << c.b << ")\n";
    }

    cv::Mat out = img.clone();
    for (const Peak& p : circles) {
        const cv::Point ctr(static_cast<int>(std::lround(p.a)), static_cast<int>(std::lround(p.b)));
        cv::circle(out, ctr, static_cast<int>(std::lround(p.r)), cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
        cv::circle(out, ctr, 3, cv::Scalar(255, 255, 0), -1, cv::LINE_AA);
    }

    cv::imwrite((outDir / "0_kenar_resmi_ince.png").string(), thin);
    cv::imwrite((outDir / "1_oylama_yaricap_boyunca_en_buyuk.png").string(), mipColor);
    if (!sliceImg.empty()) cv::imwrite((outDir / "2_oylama_dilimi_en_guclu_cember.png").string(), sliceImg);
    cv::imwrite((outDir / "3_uc_pikselin_oy_cemberleri.png").string(), demo);
    cv::imwrite((outDir / "4_bulunan_cemberler_camgobegi.png").string(), out);

    std::cout << "\nKaydedilenler (" << outDir.string() << "):\n"
              << "  0_kenar_resmi_ince.png                     : oylamada kullanilan inceltilmis kenar pikselleri\n"
              << "  1_oylama_yaricap_boyunca_en_buyuk.png      : 3 boyutlu matrisin yaricap boyunca en buyuk deger izdusumu (parlak noktalar = merkez adaylari)\n"
              << "  2_oylama_dilimi_en_guclu_cember.png        : en guclu cemberin yaricap diliminin oylari (kirmizi daire = secilen merkez)\n"
              << "  3_uc_pikselin_oy_cemberleri.png            : ayni cember uzerindeki 3 pikselin oy cemberleri tek noktada kesisir\n"
              << "  4_bulunan_cemberler_camgobegi.png          : Madde 4 sonucu: bulunan cemberler orijinal resim uzerinde (camgobegi)\n";
    return 0;
}
