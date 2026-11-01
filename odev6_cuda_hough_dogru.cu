// Ödev 6 - Hough doğru dönüşümünün CUDA (GPU) ile paralel gerçeklenmesi ve testi.
//
// Oylama (her kenar pikselinin her theta için rho hesaplayıp oy vermesi) GPU'da paralel yapılır.
// İki GPU çekirdeği yazıldı ve karşılaştırıldı:
//   1) "naif" çekirdek: her (piksel, theta) çifti için ayrı bir GPU thread'i, oylar doğrudan global belleğe
//      atomik toplamayla yazılır (atomicAdd). Çok sayıda thread aynı hücreye oy verince çakışma yavaşlatır.
//   2) "paylaşılan bellek" çekirdeği: her theta için bir blok. Bloğun rho sütunu (2001 sayı) hızlı paylaşılan
//      bellekte tutulur, bloktaki thread'ler kenar piksellerini paylaşıp oyları orada toplar; sonunda sütun
//      global belleğe tek seferde yazılır. Global bellekte atomik işlem yoktur.
// Oylama matrisi theta-ana düzenindedir: acc[theta * rhoSayisi + rho].
//
// Test: CPU (1 thread ve çok thread) oylaması ile GPU oylama matrisleri hücre hücre karşılaştırılır
// (birebir aynı olmalı), süreler ölçülür, GPU matrisinden doğrular bulunup resme çizilir.
//
// Kullanım: ./cuda_hough <goruntu> [cikti_dizini] [esik_orani=0.4] [theta_adim_derece=1.0] [tekrar=30]
// Derleme (Google Colab, GPU çalışma zamanı; OpenCV: apt-get install libopencv-dev):
//   nvcc -O2 -std=c++17 --fmad=false -arch=native odev6_cuda_hough_dogru.cu -o cuda_hough \
//        $(pkg-config --cflags --libs opencv4) -Xcompiler "-O2,-pthread"
// (hough_ortak.hpp bu dosyayla aynı klasörde olmalıdır; --fmad=false CPU ile aynı ondalık sonucu verir)

#include "hough_ortak.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                                                  \
    do {                                                                                                  \
        cudaError_t err__ = (call);                                                                       \
        if (err__ != cudaSuccess) {                                                                       \
            std::cerr << "CUDA hatasi: " << cudaGetErrorString(err__) << " (satir " << __LINE__ << ")\n"; \
            std::exit(1);                                                                                 \
        }                                                                                                 \
    } while (0)

static const double PI = 3.14159265358979323846;

// ---------------------------------------------------------------------------------------------
// GPU çekirdekleri
// ---------------------------------------------------------------------------------------------

// 1) Naif: her thread bir (piksel, theta) çiftini işler, oyu global belleğe atomik ekler
__global__ void houghNaiveKernel(const int* xs, const int* ys, int n, const double* cosT, const double* sinT, int numTheta, int D, int numRho, int* acc) {
    const long long total = static_cast<long long>(n) * numTheta;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total; idx += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int i = static_cast<int>(idx / numTheta);
        const int t = static_cast<int>(idx % numTheta);
        const double rho = xs[i] * cosT[t] + ys[i] * sinT[t];
        const int r = static_cast<int>(lround(rho)) + D;
        atomicAdd(&acc[static_cast<size_t>(t) * numRho + r], 1);
    }
}

// 2) Paylaşılan bellek: bir blok = bir theta. Blok kendi rho sütununu paylaşılan bellekte biriktirir.
__global__ void houghSharedKernel(const int* xs, const int* ys, int n, const double* cosT, const double* sinT, int D, int numRho, int* acc) {
    extern __shared__ int col[];  // numRho sayi
    const int t = blockIdx.x;
    for (int r = threadIdx.x; r < numRho; r += blockDim.x) col[r] = 0;
    __syncthreads();

    const double c = cosT[t], s = sinT[t];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const double rho = xs[i] * c + ys[i] * s;
        const int r = static_cast<int>(lround(rho)) + D;
        atomicAdd(&col[r], 1);  // paylasilan bellekte atomik toplama (global bellege gore cok hizli)
    }
    __syncthreads();

    for (int r = threadIdx.x; r < numRho; r += blockDim.x) acc[static_cast<size_t>(t) * numRho + r] = col[r];  // sutunu tek seferde yaz
}

// ---------------------------------------------------------------------------------------------
// CPU oylaması (1 thread veya çok thread; theta aralıklara bölünür, her thread kendi sütunlarına yazar)
// ---------------------------------------------------------------------------------------------
static void houghCpu(const std::vector<int>& xs, const std::vector<int>& ys, const std::vector<double>& cosT, const std::vector<double>& sinT, int D,
                     int numRho, int numTheta, std::vector<int>& acc, int threads) {
    std::fill(acc.begin(), acc.end(), 0);
    auto work = [&](int tb, int te) {
        for (int t = tb; t < te; ++t) {
            const double c = cosT[t], s = sinT[t];
            int* colp = &acc[static_cast<size_t>(t) * numRho];
            for (size_t i = 0; i < xs.size(); ++i) {
                const double rho = xs[i] * c + ys[i] * s;
                ++colp[static_cast<int>(std::lround(rho)) + D];
            }
        }
    };
    if (threads <= 1) {
        work(0, numTheta);
        return;
    }
    std::vector<std::thread> workers;
    const int chunk = (numTheta + threads - 1) / threads;
    for (int k = 1; k < threads; ++k) {
        const int b = k * chunk, e = std::min(numTheta, b + chunk);
        if (b < e) workers.emplace_back(work, b, e);
    }
    work(0, std::min(numTheta, chunk));
    for (auto& w : workers) w.join();
}

template <typename Fn>
static double averageMs(Fn fn, int repeat) {
    fn();  // ısınma
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) fn();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / repeat;
}

// ---------------------------------------------------------------------------------------------
// Tepe bulma (Madde 2'deki CPU programıyla aynı yöntem) ve yardımcılar
// ---------------------------------------------------------------------------------------------
struct Peak {
    double rho, theta;  // theta derece
    int votes;
};

static std::vector<Peak> findLines(const std::vector<int>& acc, int D, int numRho, int numTheta, double stepDeg, double ratio, int& threshold, int& maxVotes) {
    maxVotes = *std::max_element(acc.begin(), acc.end());
    threshold = std::max(1, static_cast<int>(std::lround(ratio * maxVotes)));
    const int lowVotes = std::max(1, maxVotes / 10);
    const int halfRho = 10, halfTheta = std::max(1, static_cast<int>(std::lround(6.0 / stepDeg)));
    const int cw = std::max(1, static_cast<int>(std::lround(3.0 / stepDeg)));

    auto at = [&](int r, int t) -> int {
        if (t < 0) {
            t += numTheta;
            r = 2 * D - r;
        } else if (t >= numTheta) {
            t -= numTheta;
            r = 2 * D - r;
        }
        if (r < 0 || r >= numRho) return 0;
        return acc[static_cast<size_t>(t) * numRho + r];
    };

    std::vector<Peak> peaks;
    for (int t = 0; t < numTheta; ++t) {
        for (int r = 0; r < numRho; ++r) {
            const int v = acc[static_cast<size_t>(t) * numRho + r];
            if (v < lowVotes) continue;
            bool isPeak = true;
            for (int dr = -halfRho; dr <= halfRho && isPeak; ++dr)
                for (int dt = -halfTheta; dt <= halfTheta; ++dt) {
                    if (dr == 0 && dt == 0) continue;
                    const int nv = at(r + dr, t + dt);
                    if (nv > v || (nv == v && (dr < 0 || (dr == 0 && dt < 0)))) {
                        isPeak = false;
                        break;
                    }
                }
            if (!isPeak) continue;
            double sw = 0, sr = 0, st = 0;
            for (int dr = -4; dr <= 4; ++dr)
                for (int dt = -cw; dt <= cw; ++dt) {
                    const int w = at(r + dr, t + dt);
                    if (w * 2 >= v) {
                        sw += w;
                        sr += static_cast<double>(w) * (r + dr - D);
                        st += static_cast<double>(w) * (t + dt);
                    }
                }
            peaks.push_back({sr / sw, st / sw * stepDeg, v});
        }
    }
    std::sort(peaks.begin(), peaks.end(), [](const Peak& a, const Peak& b) { return a.votes > b.votes; });
    return peaks;
}

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

static double footDistance(double r1, double t1, double r2, double t2) {
    const double a1 = t1 * PI / 180.0, a2 = t2 * PI / 180.0;
    return std::hypot(r1 * std::cos(a1) - r2 * std::cos(a2), r1 * std::sin(a1) - r2 * std::sin(a2));
}

static cv::Mat accToImage(const std::vector<int>& acc, int numRho, int numTheta, int scaleX, double maxVal) {
    cv::Mat img(numRho, numTheta * scaleX, CV_8UC1);
    for (int r = 0; r < numRho; ++r) {
        uchar* row = img.ptr<uchar>(r);
        for (int t = 0; t < numTheta; ++t) {
            const double v = maxVal > 0 ? std::sqrt(acc[static_cast<size_t>(t) * numRho + r] / maxVal) : 0.0;
            const uchar g = cv::saturate_cast<uchar>(v * 255.0);
            for (int k = 0; k < scaleX; ++k) row[t * scaleX + k] = g;
        }
    }
    return img;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Kullanim: " << argv[0] << " <goruntu> [cikti_dizini] [esik_orani=0.4] [theta_adim_derece=1.0] [tekrar=30]\n";
        return 1;
    }
    const std::string inPath = argv[1];
    const fs::path outDir = (argc >= 3) ? fs::path(argv[2]) : fs::path("cikti_cuda_hough");
    const double ratio = (argc >= 4) ? std::stod(argv[3]) : 0.4;
    const double stepDeg = (argc >= 5) ? std::stod(argv[4]) : 1.0;
    const int repeat = (argc >= 6) ? std::max(1, std::stoi(argv[5])) : 30;
    fs::create_directories(outDir);

    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount == 0) {
        std::cerr << "Hata: GPU bulunamadi. Colab'da Calisma zamani turunu GPU (T4) yapin.\n";
        return 1;
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const cv::Mat img = cv::imread(inPath, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Hata: goruntu okunamadi: " << inPath << "\n";
        return 1;
    }
    const hough::Edges e = hough::makeEdges(img);
    const int W = img.cols, H = img.rows;

    std::vector<int> xs, ys;
    for (int y = 0; y < H; ++y) {
        const uchar* row = e.cleaned.ptr<uchar>(y);
        for (int x = 0; x < W; ++x)
            if (row[x]) {
                xs.push_back(x);
                ys.push_back(y);
            }
    }
    const int n = static_cast<int>(xs.size());
    const int D = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(W) * W + static_cast<double>(H) * H)));
    const int numRho = 2 * D + 1;
    const int numTheta = static_cast<int>(std::lround(180.0 / stepDeg));
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    const int mt = std::max(2, hw);

    std::vector<double> cosT(numTheta), sinT(numTheta);
    for (int t = 0; t < numTheta; ++t) {
        const double a = t * stepDeg * PI / 180.0;
        cosT[t] = std::cos(a);
        sinT[t] = std::sin(a);
    }

    std::cout << "GPU          : " << prop.name << " (hesaplama yetenegi " << prop.major << "." << prop.minor << ")\n";
    std::cout << "CPU thread   : " << hw << " (cok thread'li olcum " << mt << " thread ile)\n";
    std::cout << "Goruntu      : " << W << "x" << H << ", kenar piksel sayisi: " << n << "\n";
    std::cout << "Oylama matrisi: " << numRho << " (rho) x " << numTheta << " (theta, adim " << stepDeg << " derece) = " << static_cast<double>(numRho) * numTheta * 4 / 1e6
              << " MB; toplam oy = " << static_cast<double>(n) * numTheta << "\n\n";

    // ---- GPU belleği ----
    int *d_xs, *d_ys, *d_acc;
    double *d_cos, *d_sin;
    const size_t accCells = static_cast<size_t>(numRho) * numTheta;
    CUDA_CHECK(cudaMalloc(&d_xs, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ys, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cos, numTheta * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sin, numTheta * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_acc, accCells * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_cos, cosT.data(), numTheta * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, sinT.data(), numTheta * sizeof(double), cudaMemcpyHostToDevice));

    const int shBytes = numRho * static_cast<int>(sizeof(int));
    auto uploadPixels = [&] {
        CUDA_CHECK(cudaMemcpy(d_xs, xs.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, ys.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    };
    auto runNaive = [&] {
        CUDA_CHECK(cudaMemset(d_acc, 0, accCells * sizeof(int)));
        houghNaiveKernel<<<4096, 256>>>(d_xs, d_ys, n, d_cos, d_sin, numTheta, D, numRho, d_acc);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    auto runShared = [&] {
        houghSharedKernel<<<numTheta, 256, shBytes>>>(d_xs, d_ys, n, d_cos, d_sin, D, numRho, d_acc);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    std::vector<int> accCpu1(accCells), accCpuN(accCells), accNaive(accCells), accShared(accCells);

    // ---- DOĞRULUK: oylama matrislerini hücre hücre karşılaştır ----
    houghCpu(xs, ys, cosT, sinT, D, numRho, numTheta, accCpu1, 1);
    houghCpu(xs, ys, cosT, sinT, D, numRho, numTheta, accCpuN, mt);
    uploadPixels();
    runNaive();
    CUDA_CHECK(cudaMemcpy(accNaive.data(), d_acc, accCells * sizeof(int), cudaMemcpyDeviceToHost));
    runShared();
    CUDA_CHECK(cudaMemcpy(accShared.data(), d_acc, accCells * sizeof(int), cudaMemcpyDeviceToHost));

    auto diffCells = [&](const std::vector<int>& a, const std::vector<int>& b) {
        long long d = 0;
        for (size_t i = 0; i < a.size(); ++i) d += (a[i] != b[i]) ? 1 : 0;
        return d;
    };
    std::cout << "DOGRULUK (CPU 1 thread oylama matrisine gore farkli hucre sayisi, 0 = birebir ayni):\n";
    std::cout << "  CPU " << mt << " thread          : " << diffCells(accCpu1, accCpuN) << "\n";
    std::cout << "  GPU naif cekirdek       : " << diffCells(accCpu1, accNaive) << "\n";
    std::cout << "  GPU paylasilan bellek   : " << diffCells(accCpu1, accShared) << "\n\n";

    // ---- SÜRE ----
    const double msCpu1 = averageMs([&] { houghCpu(xs, ys, cosT, sinT, D, numRho, numTheta, accCpu1, 1); }, repeat);
    const double msCpuN = averageMs([&] { houghCpu(xs, ys, cosT, sinT, D, numRho, numTheta, accCpuN, mt); }, repeat);
    const double msNaive = averageMs(runNaive, repeat);
    const double msShared = averageMs(runShared, repeat);
    const double msSharedTotal = averageMs(
        [&] {
            uploadPixels();
            runShared();
            CUDA_CHECK(cudaMemcpy(accShared.data(), d_acc, accCells * sizeof(int), cudaMemcpyDeviceToHost));
        },
        repeat);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "SURE (oylama, ms, " << repeat << " tekrarin ortalamasi):\n";
    std::cout << "  CPU 1 thread                                   : " << std::setw(9) << msCpu1 << "\n";
    std::cout << "  CPU " << mt << " thread                                   : " << std::setw(9) << msCpuN << "\n";
    std::cout << "  GPU naif (global atomik), sadece cekirdek      : " << std::setw(9) << msNaive << "\n";
    std::cout << "  GPU paylasilan bellek, sadece cekirdek         : " << std::setw(9) << msShared << "\n";
    std::cout << "  GPU paylasilan bellek, yukleme + indirme dahil : " << std::setw(9) << msSharedTotal << "\n";
    std::cout << std::setprecision(2);
    std::cout << "HIZLANMA (CPU 1 thread / GPU): naif " << msCpu1 / msNaive << "x, paylasilan bellek (cekirdek) " << msCpu1 / msShared
              << "x, paylasilan bellek (toplam) " << msCpu1 / msSharedTotal << "x\n\n";

    // ---- GPU matrisinden doğruları bul ----
    int threshold = 0, maxVotes = 0;
    const std::vector<Peak> peaks = findLines(accShared, D, numRho, numTheta, stepDeg, ratio, threshold, maxVotes);
    int thrCpu = 0, maxCpu = 0;
    const std::vector<Peak> peaksCpu = findLines(accCpu1, D, numRho, numTheta, stepDeg, ratio, thrCpu, maxCpu);
    std::vector<Peak> lines;
    for (const Peak& p : peaks)
        if (p.votes >= threshold) lines.push_back(p);
    int linesCpu = 0;
    for (const Peak& p : peaksCpu) linesCpu += (p.votes >= thrCpu) ? 1 : 0;

    std::cout << std::setprecision(1);
    std::cout << "GPU oylama matrisinden doguru bulma: en yuksek oy = " << maxVotes << ", esik = " << ratio << " x " << maxVotes << " = " << threshold << "\n";
    std::cout << "Esigi gecen dogru sayisi: GPU " << lines.size() << ", CPU " << linesCpu << "\n";
    for (const Peak& p : lines) {
        double rho = p.rho, th = p.theta;
        canonical(rho, th);
        std::cout << "  rho = " << std::setw(7) << rho << ", theta = " << std::setw(6) << th << " derece, oy = " << p.votes << "\n";
    }

    if (inPath.find("sahne") != std::string::npos) {
        const double truth[4][2] = {{60.0, 90.0}, {740.0, 0.0}, {410.1, 45.0}, {670.8, 63.4}};
        std::cout << "\nGERCEK DOGRULARLA KARSILASTIRMA (sahne.png):\n";
        for (const auto& tr : truth) {
            double bestD = 1e18;
            int best = -1;
            for (size_t i = 0; i < lines.size(); ++i) {
                const double d = footDistance(tr[0], tr[1], lines[i].rho, lines[i].theta);
                if (d < bestD) {
                    bestD = d;
                    best = static_cast<int>(i);
                }
            }
            if (best < 0) {
                std::cout << "  gercek (rho = " << tr[0] << ", theta = " << tr[1] << "): BULUNAMADI\n";
            } else {
                double rho = lines[best].rho, th = lines[best].theta;
                canonical(rho, th);
                std::cout << "  gercek (rho = " << std::setw(6) << tr[0] << ", theta = " << std::setw(5) << tr[1] << ") -> bulunan (rho = " << std::setw(7) << rho
                          << ", theta = " << std::setw(6) << th << "), orijine en yakin noktalar arasi = " << bestD << " piksel\n";
            }
        }
    }

    // ---- Görseller ----
    cv::Mat out = img.clone();
    for (const Peak& p : lines) hough::drawLineRhoTheta(out, p.rho, p.theta * PI / 180.0, cv::Scalar(0, 255, 0), 2);
    cv::imwrite((outDir / "1_gpu_oylama_matrisi.png").string(), accToImage(accShared, numRho, numTheta, std::max(1, static_cast<int>(std::lround(4.0 * stepDeg))), maxVotes));
    cv::imwrite((outDir / "2_gpu_ile_bulunan_dogrular_yesil.png").string(), out);

    cudaFree(d_xs);
    cudaFree(d_ys);
    cudaFree(d_cos);
    cudaFree(d_sin);
    cudaFree(d_acc);

    std::cout << "\nKaydedilenler (" << outDir.string() << "):\n"
              << "  1_gpu_oylama_matrisi.png            : GPU'da hesaplanan oylama matrisi\n"
              << "  2_gpu_ile_bulunan_dogrular_yesil.png: GPU oylamasindan bulunan dogrular (yesil)\n";
    return 0;
}
