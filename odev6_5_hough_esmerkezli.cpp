// Ödev 6 - Madde 5: Eş merkezli çemberleri Hough oylamasıyla bulma (hazır fonksiyon yok, elle yazılmış).
//
// Fikir: Eş merkezli çemberler aynı merkezde farklı yarıçaplı birden çok HALKA'dır. Oylamanın kendisi
// "bu merkezde kaç halka onaylandı?" sorusunu cevaplayacak şekilde tasarlandı; çemberler önce bulunup
// sonra merkezlerine göre gruplanmıyor.
//
// Oylama iki katmanlıdır ve ikisi de oylama döngüsünün İÇİNDE çalışır:
//   1. katman: 3 boyutlu matris A(r, b, a). Madde 4'teki gibi, her kenar pikseli her yarıçap diliminde
//              merkez adaylarına oy verir (sin/cos tablosu bir kere hesaplanır, belleğe alınır).
//   2. katman: 2 boyutlu "halka sayısı" matrisi C(a, b). Yarıçap dilimleri küçükten büyüğe sırayla
//              oylanırken, bir dilimin oylaması bittiğinde (oylar 3x3 komşulukta toplanır) A(r, b, a) skoru
//              eşiği geçen her (a, b) hücresi o yarıçaptaki bir HALKA'yı onaylar ve C(a, b) bir artar.
//              Aynı halka birkaç komşu yarıçap diliminde eşiği geçebilir; hücrenin son onayından itibaren
//              "halka aralığı" kadar yarıçap içindeki onaylar aynı halkanın devamı sayılır, C artmaz
//              (yalnızca o halkanın en güçlü dilimi hatırlanır).
//   Sonuç: C(a, b) = o merkezdeki eş merkezli halka sayısı. C >= min_halka olan merkezler eş merkezli gruplardır;
//          halkaların yarıçapları oylama sırasında kaydedildiği için ayrıca arama yapılmaz.
//
// Skor = oy / (2*pi*r): "çemberin çevresinin ne kadarı kenar pikselleriyle destekleniyor" (Madde 4'teki gibi).
// Eşik mutlak skor olarak verilir, çünkü oylama dilim dilim ilerlerken en yüksek skor henüz bilinmez.
// Varsayılan 3.4, Madde 4 çıktısına göre seçildi (gerçek çemberler 4.06-4.52, sahte tepeler en çok 2.66).
//
// Kullanım: ./hough_esmerkezli <goruntu> [cikti_dizini] [esik_skor=3.4] [min_halka=2] [r_min=30] [r_max=140]
// Derleme:  g++ -O2 odev6_5_hough_esmerkezli.cpp -o hough_esmerkezli.exe -std=c++17 $(pkg-config --cflags --libs opencv5)
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

struct Ring {
    double r, a, b;
    int votes;
};
struct Group {
    double a, b;
    int sumVotes;
    std::vector<Ring> rings;
};

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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Kullanim: " << argv[0] << " <goruntu> [cikti_dizini] [esik_skor=3.4] [min_halka=2] [r_min=30] [r_max=140]\n";
        return 1;
    }
    const std::string inPath = argv[1];
    const fs::path outDir = (argc >= 3) ? fs::path(argv[2]) : fs::path("cikti_hough_esmerkezli");
    const double scoreThreshold = (argc >= 4) ? std::stod(argv[3]) : 3.4;
    const int minRings = (argc >= 5) ? std::max(1, std::stoi(argv[4])) : 2;
    const int rMin = (argc >= 6) ? std::stoi(argv[5]) : 30;
    const int rMax = (argc >= 7) ? std::stoi(argv[6]) : 140;
    const int ringGap = 7;  // bu kadar yaricap icindeki onaylar ayni halkanin devami sayilir
    const int K = 8;        // bir merkezde izlenebilecek en fazla halka sayisi
    fs::create_directories(outDir);

    const cv::Mat img = cv::imread(inPath, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Hata: goruntu okunamadi: " << inPath << "\n";
        return 1;
    }
    const int W = img.cols, H = img.rows;
    const hough::Edges e = hough::makeEdges(img);
    cv::Mat thin;
    cv::erode(e.cleaned, thin, cv::Mat::ones(3, 3, CV_8UC1));
    std::vector<cv::Point> pts;
    for (int y = 0; y < H; ++y) {
        const uchar* row = thin.ptr<uchar>(y);
        for (int x = 0; x < W; ++x)
            if (row[x]) pts.emplace_back(x, y);
    }

    const int numTheta = 1440;
    const int nR = rMax - rMin + 1;
    const size_t cells = static_cast<size_t>(W) * H;
    std::cout << "Goruntu boyutu: " << W << "x" << H << ", oylamada kullanilan kenar piksel sayisi: " << pts.size() << "\n";
    std::cout << "3 boyutlu oylama matrisi A(r, b, a): yaricap " << rMin << ".." << rMax << " (" << nR << " dilim), "
              << static_cast<double>(nR) * cells * sizeof(uint16_t) / 1e6 << " MB; 2 boyutlu halka sayisi matrisi C(a, b)\n";
    std::cout << "Esik skor = " << scoreThreshold << " (oy / (2*pi*r)), minimum halka sayisi = " << minRings << ", halka araligi = " << ringGap << "\n\n";

    // sin ve cos tablosu: bir kere hesaplanir, bellekte tutulur
    std::vector<double> cosT(numTheta), sinT(numTheta);
    for (int t = 0; t < numTheta; ++t) {
        const double a = t * 2.0 * PI / numTheta;
        cosT[t] = std::cos(a);
        sinT[t] = std::sin(a);
    }

    std::vector<uint16_t> acc(static_cast<size_t>(nR) * cells, 0);
    std::vector<uint8_t> ringCount(cells, 0);       // C(a, b): halka sayisi
    std::vector<int16_t> lastR(cells, -1000);       // hucrenin son onay aldigi yaricap
    std::vector<int16_t> ringBinR(cells * K, 0);    // her halkanin en guclu yaricap dilimi (indeks)
    std::vector<uint16_t> ringBinV(cells * K, 0);   // o dilimdeki oy
    std::vector<int> dx, dy, tmp(cells);

    std::cout << "OYLAMA: yaricap dilimleri kucukten buyuge sirayla oylaniyor; bir dilim bitince esigi gecen hucreler halka onaylar:\n";
    int events = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int ri = 0; ri < nR; ++ri) {
        const int r = rMin + ri;
        const int n = std::max(8, static_cast<int>(std::lround(2.0 * PI * r)));
        dx.assign(n, 0);
        dy.assign(n, 0);
        for (int i = 0; i < n; ++i) {
            const int t = static_cast<int>(static_cast<long long>(i) * numTheta / n);
            dx[i] = static_cast<int>(std::lround(r * cosT[t]));
            dy[i] = static_cast<int>(std::lround(r * sinT[t]));
        }

        // 1. katman: bu yaricap icin 3 boyutlu matrisin dilimine oy ver
        uint16_t* slice = &acc[static_cast<size_t>(ri) * cells];
        for (const cv::Point& p : pts) {
            for (int i = 0; i < n; ++i) {
                const int a = p.x - dx[i];
                const int b = p.y - dy[i];
                if (static_cast<unsigned>(a) < static_cast<unsigned>(W) && static_cast<unsigned>(b) < static_cast<unsigned>(H))
                    ++slice[static_cast<size_t>(b) * W + a];
            }
        }
        boxFilter3x3(slice, W, H, tmp);

        // 2. katman: esigi gecen hucreler bu yaricapta bir HALKA onaylar, C(a, b) artar
        const int thrVotes = static_cast<int>(std::ceil(scoreThreshold * 2.0 * PI * r));
        for (size_t idx = 0; idx < cells; ++idx) {
            const int v = slice[idx];
            if (v < thrVotes) continue;
            const int cnt = ringCount[idx];
            if (cnt > 0 && r - lastR[idx] <= ringGap) {  // ayni halkanin devami: en guclu dilimi sakla, C artmaz
                if (v > ringBinV[idx * K + cnt - 1]) {
                    ringBinV[idx * K + cnt - 1] = static_cast<uint16_t>(v);
                    ringBinR[idx * K + cnt - 1] = static_cast<int16_t>(ri);
                }
                lastR[idx] = static_cast<int16_t>(r);
            } else if (cnt < K) {  // yeni halka: C(a, b) += 1
                ringBinV[idx * K + cnt] = static_cast<uint16_t>(v);
                ringBinR[idx * K + cnt] = static_cast<int16_t>(ri);
                lastR[idx] = static_cast<int16_t>(r);
                ringCount[idx] = static_cast<uint8_t>(cnt + 1);
                if (events < 12) {
                    std::cout << "  r = " << std::setw(3) << r << ": hucre (a = " << std::setw(3) << idx % W << ", b = " << std::setw(3) << idx / W
                              << ") esigi gecti (oy = " << v << ", skor = " << std::fixed << std::setprecision(2) << v / (2.0 * PI * r)
                              << ") -> yeni halka onaylandi, C(a, b) = " << cnt + 1 << "\n";
                    ++events;
                }
            }
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    std::cout << "  (ilk 12 onay gosterildi)\nOylama tamamlandi, sure = " << std::fixed << std::setprecision(1)
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    int maxC = 0;
    long long cellsWithRing = 0;
    for (size_t i = 0; i < cells; ++i) {
        maxC = std::max<int>(maxC, ringCount[i]);
        cellsWithRing += ringCount[i] > 0 ? 1 : 0;
    }
    std::cout << "Halka onayi alan hucre sayisi: " << cellsWithRing << ", en buyuk C(a, b) = " << maxC << "\n";

    // Merkez secimi: komsu hucreler ayni halkalari onayladigi icin, 8 piksellik pencerede en iyi
    // (halka sayisi, toplam oy) degerine sahip hucre merkez olur
    std::vector<int> sumV(cells, 0);
    for (size_t i = 0; i < cells; ++i)
        for (int k = 0; k < ringCount[i]; ++k) sumV[i] += ringBinV[i * K + k];

    auto at = [&](int ri, int b, int a) -> int {
        if (ri < 0 || ri >= nR || b < 0 || b >= H || a < 0 || a >= W) return 0;
        return acc[static_cast<size_t>(ri) * cells + static_cast<size_t>(b) * W + a];
    };

    std::vector<Group> groups;
    const int win = 8;
    for (int b = 0; b < H; ++b) {
        for (int a = 0; a < W; ++a) {
            const size_t idx = static_cast<size_t>(b) * W + a;
            if (ringCount[idx] == 0) continue;
            bool best = true;
            for (int db = -win; db <= win && best; ++db) {
                for (int da = -win; da <= win; ++da) {
                    const int nb = b + db, na = a + da;
                    if ((db == 0 && da == 0) || nb < 0 || nb >= H || na < 0 || na >= W) continue;
                    const size_t n = static_cast<size_t>(nb) * W + na;
                    if (ringCount[n] == 0) continue;
                    if (ringCount[n] > ringCount[idx] || (ringCount[n] == ringCount[idx] && (sumV[n] > sumV[idx] || (sumV[n] == sumV[idx] && n < idx)))) {
                        best = false;
                        break;
                    }
                }
            }
            if (!best) continue;

            Group g;
            g.sumVotes = sumV[idx];
            double ga = 0, gb = 0;
            for (int k = 0; k < ringCount[idx]; ++k) {
                const int ri = ringBinR[idx * K + k];
                const int v = ringBinV[idx * K + k];
                double sw = 0, sa = 0, sb = 0;  // halka merkezi: komsu oylarin agirlik merkezi
                for (int db = -3; db <= 3; ++db)
                    for (int da = -3; da <= 3; ++da) {
                        const int w = at(ri, b + db, a + da);
                        if (w * 2 >= v) {
                            sw += w;
                            sa += static_cast<double>(w) * (a + da);
                            sb += static_cast<double>(w) * (b + db);
                        }
                    }
                double rw = 0, rs = 0;  // halka yaricapi: komsu yaricaplardaki oylarin agirlik ortalamasi
                for (int dri = -ringGap; dri <= ringGap; ++dri) {
                    const int w = at(ri + dri, b, a);
                    if (w * 2 >= v) {
                        rw += w;
                        rs += static_cast<double>(w) * (rMin + ri + dri);
                    }
                }
                Ring ring{rs / rw, sa / sw, sb / sw, v};
                g.rings.push_back(ring);
                ga += ring.a;
                gb += ring.b;
            }
            g.a = ga / g.rings.size();
            g.b = gb / g.rings.size();
            std::sort(g.rings.begin(), g.rings.end(), [](const Ring& x, const Ring& y) { return x.r < y.r; });
            groups.push_back(g);
        }
    }
    std::sort(groups.begin(), groups.end(), [](const Group& x, const Group& y) {
        if (x.rings.size() != y.rings.size()) return x.rings.size() > y.rings.size();
        return x.sumVotes > y.sumVotes;
    });

    std::cout << "\nBULUNAN MERKEZLER (" << groups.size() << "): halka sayisi C >= " << minRings << " olanlar ES MERKEZLI, digerleri tek cember:\n";
    int concentricCount = 0;
    for (const Group& g : groups) {
        const bool conc = static_cast<int>(g.rings.size()) >= minRings;
        concentricCount += conc ? 1 : 0;
        std::cout << "  " << (conc ? "ES MERKEZLI" : "tek cember ") << " merkez = (" << std::setprecision(1) << g.a << ", " << g.b << "), C = " << g.rings.size()
                  << ", yaricaplar:";
        for (const Ring& rg : g.rings) std::cout << " " << rg.r << " (skor " << std::setprecision(2) << rg.votes / (2.0 * PI * rg.r) << ")" << std::setprecision(1);
        std::cout << "\n";
    }
    std::cout << "Es merkezli grup sayisi: " << concentricCount << "\n";

    if (inPath.find("sahne") != std::string::npos) {
        std::cout << "\nGERCEK DEGERLER (sahne.png): es merkezli grup: merkez (560, 320), yaricaplar 40, 80, 120; tek cemberler: (200, 200, 70), (330, 430, 60)\n";
        for (const Group& g : groups) {
            if (std::hypot(g.a - 560.0, g.b - 320.0) < 5.0 && g.rings.size() >= 3)
                std::cout << "  es merkezli grup bulundu: merkez hatasi = " << std::hypot(g.a - 560.0, g.b - 320.0) << " piksel, yaricap hatalari = "
                          << std::fabs(g.rings[0].r - 40) << ", " << std::fabs(g.rings[1].r - 80) << ", " << std::fabs(g.rings[2].r - 120) << " piksel\n";
        }
    }

    // ---- Gorseller ----
    // C(a, b) matrisi: halka sayisi (ne kadar parlaksa o kadar cok halka)
    cv::Mat cMap(H, W, CV_8UC1, cv::Scalar(0));
    for (size_t i = 0; i < cells; ++i) cMap.data[i] = static_cast<uchar>(std::min(255, ringCount[i] * (maxC > 0 ? 255 / maxC : 0)));
    cv::Mat cMapDil;
    cv::dilate(cMap, cMapDil, cv::Mat::ones(5, 5, CV_8UC1));
    cv::Mat cColor = hough::toBgr(cMapDil);
    for (const Group& g : groups)
        cv::circle(cColor, cv::Point(static_cast<int>(std::lround(g.a)), static_cast<int>(std::lround(g.b))), 14,
                   static_cast<int>(g.rings.size()) >= minRings ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 160, 0), 2, cv::LINE_AA);

    // en guclu es merkezli grubun halkalarinin dilimleri (merkez etrafinda 60 piksellik kesit, 4 kat buyutulmus)
    cv::Mat panel;
    for (const Group& g : groups) {
        if (static_cast<int>(g.rings.size()) < minRings) continue;
        const int ca = static_cast<int>(std::lround(g.a)), cb = static_cast<int>(std::lround(g.b));
        int maxV = 1;
        for (const Ring& rg : g.rings) maxV = std::max(maxV, rg.votes);
        std::vector<cv::Mat> parts;
        for (const Ring& rg : g.rings) {
            const int ri = std::min(nR - 1, std::max(0, static_cast<int>(std::lround(rg.r)) - rMin));
            cv::Mat crop(121, 121, CV_8UC1, cv::Scalar(0));
            for (int dy2 = -60; dy2 <= 60; ++dy2)
                for (int dx2 = -60; dx2 <= 60; ++dx2) {
                    const double v = std::sqrt(static_cast<double>(at(ri, cb + dy2, ca + dx2)) / maxV);
                    crop.at<uchar>(dy2 + 60, dx2 + 60) = cv::saturate_cast<uchar>(v * 255.0);
                }
            cv::Mat big;
            cv::resize(crop, big, cv::Size(), 4.0, 4.0, cv::INTER_NEAREST);
            parts.push_back(big);
        }
        cv::hconcat(parts, panel);
        break;
    }

    cv::Mat out = img.clone();
    const cv::Scalar groupColors[3] = {cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(0, 200, 255)};
    int gi = 0;
    for (const Group& g : groups) {
        const bool conc = static_cast<int>(g.rings.size()) >= minRings;
        const cv::Scalar col = conc ? groupColors[gi++ % 3] : cv::Scalar(255, 0, 0);
        for (const Ring& rg : g.rings)
            cv::circle(out, cv::Point(static_cast<int>(std::lround(g.a)), static_cast<int>(std::lround(g.b))), static_cast<int>(std::lround(rg.r)), col,
                       conc ? 3 : 2, cv::LINE_AA);
        cv::circle(out, cv::Point(static_cast<int>(std::lround(g.a)), static_cast<int>(std::lround(g.b))), 4, col, -1, cv::LINE_AA);
    }

    cv::imwrite((outDir / "0_kenar_resmi_ince.png").string(), thin);
    cv::imwrite((outDir / "1_halka_sayisi_matrisi_C.png").string(), cColor);
    if (!panel.empty()) cv::imwrite((outDir / "2_ayni_merkezde_halka_dilimleri.png").string(), panel);
    cv::imwrite((outDir / "3_bulunan_esmerkezli_cemberler.png").string(), out);

    std::cout << "\nKaydedilenler (" << outDir.string() << "):\n"
              << "  0_kenar_resmi_ince.png                    : oylamada kullanilan kenar pikselleri\n"
              << "  1_halka_sayisi_matrisi_C.png              : C(a, b): her merkez adayinda onaylanan halka sayisi (parlak = cok halka)\n"
              << "                                              yesil daire = es merkezli merkez, mavi daire = tek cember merkezi\n"
              << "  2_ayni_merkezde_halka_dilimleri.png       : es merkezli grubun her halkasinin oylama dilimi, soldan saga kucuk yaricaptan buyuge\n"
              << "                                              (ayni noktada parlayan uc dilim = ayni merkeze birikmis uc halka)\n"
              << "  3_bulunan_esmerkezli_cemberler.png        : Madde 5 sonucu: es merkezli gruplar yesil/kirmizi/turuncu, tek cemberler mavi\n";
    return 0;
}
