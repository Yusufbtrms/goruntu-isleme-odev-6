# Ödev 6: Hough Dönüşümleri Raporu

Bu rapor, doğrular ve çemberler için Hough dönüşümü denemelerini özetler: OpenCV'nin hazır fonksiyonları, elle yazılmış Hough doğru, çember ve eş merkezli çember algoritmaları ve doğru Hough'unun CUDA sürümü. Tüm denemeler, doğruları ve çemberleri kesin bilinen 800x600 bir test sahnesi (`odev6_sahne.cpp`) üzerinde yapıldı; böylece sonuçlar gerçek değerlerle karşılaştırılabildi.

## 1. Test sahnesi ve kenar resmi

Sahnede 4 doğru, 2 tek çember ve eş merkezli 3 çember (merkez (560, 320), yarıçaplar 40, 80, 120) var; çizgiler 3 piksel kalınlığında, zemin açık, hafif Gauss gürültüsü eklenmiş.

Gerçek doğrular (rho, theta): (60, 90°), (740, 0°), (410.1, 45°), (670.8, 63.4°). Gerçek çemberler (merkez x, y, yarıçap): (200, 200, 70), (330, 430, 60) ve (560, 320, 40 / 80 / 120).

**Kenar resmi ve median filtre.** Talimat "Canny, sonra median filtre" idi. Canny kenarları 1 piksel kalınlığında olduğu için, bir çizgi pikselinin 3x3 komşuluğundaki 9 pikselin yalnızca 3'ü doludur ve medyan 0 çıkar. Ölçüm bunu doğruladı:

| Adım | Kenar piksel sayısı |
|---|---|
| Canny | 10083 |
| Canny sonra doğrudan median 3x3 | 3444 (yaklaşık %66'sı silindi) |
| Canny, 3x3 genişletme, median 3x3 (Hough'a verilen) | 32626 |

Bu yüzden Canny kenarları önce 3x3 genişletilip sonra median uygulandı; böylece median kenarları silmeden yalnızca pürüzleri düzeltti. Elle yazılan çember algoritmalarında oylama için bu kalın kenar resmi 3x3 aşındırılarak (erode) 13508 piksele inceltildi, çünkü kalın kenarlar oyları gereksiz yere yayar.

## 2. OpenCV Hough fonksiyonları (madde 1 ve 3)

**HoughLines (standart) ve HoughLinesP (olasılıksal).** HoughLines (eşik 280) 18 doğru verdi, bunlar dört kümeye ayrılıyor ve dört gerçek doğrunun dördünü de kapsıyor, yanlış doğru yok (örneğin gerçek (60, 90°) için 52-66 rho aralığında 5 sonuç). Fazla sonuç, çizgiler kalın olduğu için bir doğrunun oylama matrisinde birkaç yan yana tepe oluşturmasından geliyor; OpenCV'nin tepe seçimi bunları tek doğruya indirmiyor. HoughLinesP (eşik 80, en kısa uzunluk 80, boşluk 15) 41 doğru parçası verdi.

**HoughCircles.** `cv::HoughCircles` kendi içinde Canny çalıştırdığı için kenar resmi girdi olarak verildiğinde "kenarın kenarı" bulunur; bu yüzden gri resimle de denendi (param1 = 100, param2 = 30, minDist = 30, yarıçap 15..150):

| Girdi | Bulunan çember | Gerçek çember sayısı 5'ten doğru bulunan |
|---|---|---|
| Kenar resmi | 23 | 3 (tek çemberler (198.5, 200.5, 69.7) ve (333.5, 426.5, 59.8); eş merkezli en dış (558.5, 321.5, 117.0)) |
| Gri resim | 10 | 3 ((199.5, 200.5, 66.8), (330.5, 429.5, 62.5), (560.5, 319.5, 122.6)) |

Her iki durumda da eş merkezli iç çemberler (r = 40 ve 80) doğru merkezle bulunamadı; çıkan yarıçapı 80 civarı çemberlerin merkezleri yaklaşık 40 piksel kaymıştı. `HoughCircles` birbirine `minDist`'ten yakın merkezlerden yalnızca birini tuttuğu için eş merkezli çemberlerin elenmesi beklenen bir davranıştır. Geri kalan 20 (kenar resmi) ve 7 (gri) çember sahtedir; `param2` değeri bu denemede ayarlanmadı.

## 3. Elle yazılan Hough doğru (madde 2)

Oylama matrisi 2001 (rho: -1000..1000) x 180 (theta: 0..179°) boyutundadır; sin ve cos değerleri bir kere hesaplanıp tabloda tutuldu, her kenar pikseli her theta için `rho = x cos(theta) + y sin(theta)` hesaplayıp ilgili hücreye oy verdi. 32626 piksel x 180 = 5 872 680 oy 22 ms'de verildi; 360 180 hücrenin 137 785'i oy aldı, en yüksek oy 725. Eşik 0.4 x 725 = 290 oldu.

| Gerçek (rho, theta) | Bulunan (rho, theta), oy | Orijine en yakın noktalar arası |
|---|---|---|
| (60, 90°) | (58.2, 90.0°), 725 | 1.8 piksel |
| (740, 0°) | (742.8, 0.3°), 484 | 4.8 piksel |
| (410.1, 45°) | (409.3, 44.9°), 663 | 1.1 piksel |
| (670.8, 63.4°) | (671.2, 63.3°), 365 | 1.0 piksel |

Eşiği geçen doğru sayısı tam 4. Doğru olmayan en güçlü tepe 193 oy aldı, yani 194 ile 364 arasındaki her eşik aynı dört doğruyu verir (eşik oranı 0.27 ile 0.50 arası). Açı hataları 0.0-0.3°, konum hataları 1.0-4.8 pikseldir. En büyük sapma dikey doğrudadır; olası sebep, 3 piksel kalın çizilen doğruların Canny'de iki kenar vermesi (OpenCV'de de aynı doğru için 58 ve 62 gibi iki tepe görüldü) ve algoritmanın bu iki kenardan birini seçmesidir. Oylama matrisi, oylamanın %1, %10 ve %100 anlarındaki halleri ve aynı doğru üzerindeki üç pikselin oy eğrilerinin (sinüzoidler) tek noktada kesişmesi resim olarak kaydedildi.

## 4. Elle yazılan Hough çember (madde 4)

3 boyutlu oylama matrisi (yarıçap, merkez y, merkez x), yarıçap 30..140 (111 dilim) x 600 x 800, 106.6 MB. Her yarıçapta oy çemberi, komşu oylar arasında yaklaşık 1 piksel yay aralığı kalacak şekilde (yaklaşık 2πr örnek) sin/cos tablosundan örneklendi; 13508 kenar piksel için 800 781 256 oy 724 ms'de hesaplandı. Oylar 3x3 komşulukta toplandı ve skor oy / (2πr) olarak (çemberin çevresinin ne kadarının kenarla desteklendiği) normalleştirildi.

| Gerçek | Bulunan | Skor | Merkez hatası | Yarıçap hatası |
|---|---|---|---|---|
| (200, 200, 70) | (200.2, 200.2, 69.6) | 4.36 | 0.2 | 0.4 |
| (330, 430, 60) | (330.1, 429.9, 59.6) | 4.23 | 0.1 | 0.4 |
| (560, 320, 40) | (560.0, 320.0, 40.1) | 4.06 | 0.0 | 0.1 |
| (560, 320, 80) | (559.8, 320.2, 80.2) | 4.52 | 0.3 | 0.2 |
| (560, 320, 120) | (559.7, 319.9, 119.7) | 4.44 | 0.3 | 0.3 |

Eşiği geçen tam 5 çember var; en yüksek sahte tepenin skoru 2.66'dır, yani 0.59 ile 0.89 arasındaki eşik oranları (en yüksek skor 4.52'ye göre) aynı sonucu verir. Eş merkezli üçlü, 3 boyutlu oylamada ayrı yarıçap dilimlerinde ayrı tepeler olarak zaten bulunur.

**Bir hata ve düzeltmesi.** İlk sürümde her yarıçapta sabit 360 oy örneği kullanıldı; küçük yarıçaplarda oylar küçük çevreye çok sık düştüğü için yapay olarak şiştiler. Sonuç olarak eşiği geçen 960 çember çıktı. Sahte tepelerin çoğu, 80 ve 120 yarıçaplı iki halka arasına sığan ve ikisine teğet olan küçük (yarıçap yaklaşık 22) çemberlerdi: merkezleri eş merkezli sistemin merkezinden 99-101 piksel uzaktaydı. Oy örneklemesini yarıçapla orantılı yapıp skoru çevreye göre normalleştirmek ve halka çiftlerini birleştirmek için tepe penceresini yarıçapta genişletmek bu sorunu çözdü.

## 5. Elle yazılan eş merkezli çember Hough'u (madde 5)

Oylama iki katmanlı ve ikisi de oylama döngüsünün içinde. 1. katman, madde 4'teki 3 boyutlu matristir; yarıçap dilimleri küçükten büyüğe sırayla oylanır. 2. katman, 2 boyutlu "halka sayısı" matrisi C(a, b)'dir: bir dilimin oylaması bitince skoru eşiği (3.4) geçen her (a, b) hücresi o yarıçapta bir halka onaylar ve C(a, b) bir artar. Aynı halkanın komşu dilimlerdeki onayları, hücrenin son onayından itibaren 7 yarıçap içindeyse aynı halkanın devamı sayılır (C artmaz, yalnızca halkanın en güçlü dilimi hatırlanır). Böylece her merkezdeki halka sayısı ve halkaların yarıçapları oylama sırasında birikir; çemberler önce bulunup sonra merkezlerine göre gruplanmaz.

Sonuç (oylama 792 ms, 13508 kenar piksel):

| Merkez | Halka sayısı C | Bulunan yarıçaplar (skor) | Gerçek |
|---|---|---|---|
| (559.8, 320.0) | 3 | 40.1 (4.35), 80.2 (4.62), 119.7 (4.53) | (560, 320): 40, 80, 120 |
| (200.2, 200.2) | 1 | 69.6 (4.51) | (200, 200): 70 |
| (330.1, 429.9) | 1 | 59.6 (4.40) | (330, 430): 60 |

Eş merkezli grubun merkez hatası 0.2 piksel, yarıçap hataları 0.1, 0.2 ve 0.3 pikseldir. Eşiği geçen toplam 59 hücre vardı ve hepsi bu üç merkezin çevresindeydi; sahte merkez çıkmadı. Eşik skoru (3.4), madde 4 çıktısına bakılarak seçildi; başka bir görüntüde ayarlanması gerekir.

Not: "oylama sırasında eş merkezlilik" ifadesi bu tasarımla yorumlandı; hocanın beklediği yöntemle uyuştuğu ayrıca teyit edilmedi.

## 6. CUDA ile Hough doğru

Kod `odev6_cuda_hough_dogru.cu`, ekran çıktısı `cuda_hough_sonuclari.txt`, görüntüler `cikti_cuda_hough/` klasöründedir.

**Yöntem.** Hough doğru oylaması (her kenar pikselinin her theta için rho hesaplayıp oy vermesi) GPU'da paralel yapıldı, iki çekirdekle:

- **Naif çekirdek:** her (piksel, theta) çifti için ayrı bir GPU thread'i; oy doğrudan global belleğe atomik toplamayla (`atomicAdd`) yazılır.
- **Paylaşılan bellek çekirdeği:** her theta için bir blok (180 blok, 256 thread). Bloğun rho sütunu (2001 sayı) hızlı paylaşılan bellekte biriktirilir, thread'ler kenar piksellerini paylaşır; sütun sonunda global belleğe tek seferde yazılır (global bellekte atomik işlem yoktur).

sin ve cos tabloları bir kere hesaplanıp GPU'ya kopyalandı. Oylama matrisi theta-ana düzendedir. Kenar resmi (Canny ve median) ve tepe bulma CPU'da yapılır; GPU yalnızca oylamayı yapar. Derleme: `nvcc -O2 --fmad=false` (çarp-topla kaynaştırmasını kapatmak, ondalıklı hesapların CPU ile aynı sırada yapılmasını sağlar).

**Ortam.** Google Colab, Tesla T4 (hesaplama yeteneği 7.5), 2 vCPU. Sahne: sahne.png (800x600), 32626 kenar piksel (yerel bilgisayardaki sonuçla aynı), oylama matrisi 2001 (rho) x 180 (theta, 1 derece adım) = 1.44 MB, toplam 5 872 680 oy. Süreler 30 tekrarın ortalamasıdır.

**Doğruluk** (CPU 1 thread oylama matrisine göre farklı hücre sayısı; 0 = birebir aynı):

| Karşılaştırılan | Farklı hücre sayısı |
|---|---|
| CPU 2 thread | 0 |
| GPU naif çekirdek | 0 |
| GPU paylaşılan bellek çekirdeği | 0 |

**Süre** (oylama, ms):

| Yöntem | Süre (ms) | Hızlanma (CPU 1 thread / yöntem) |
|---|---|---|
| CPU 1 thread | 47.766 | 1.00x |
| CPU 2 thread | 38.954 | 1.23x |
| GPU naif, yalnız çekirdek | 1.066 | 44.81x |
| GPU paylaşılan bellek, yalnız çekirdek | 1.158 | 41.25x |
| GPU paylaşılan bellek, veri yükleme ve indirme dahil | 1.691 | 28.24x |

**Bulunan doğrular.** GPU oylama matrisinden (eşik oranı 0.4, en yüksek oy 725, eşik 290) 4 doğru bulundu; CPU matrisinden de 4. Sonuçlar Madde 2'dekilerle (yerel bilgisayar, farklı OpenCV sürümü) aynıdır:

| Gerçek (rho, theta) | GPU matrisinden bulunan (rho, theta) | Orijine en yakın noktalar arası |
|---|---|---|
| (60, 90°) | (58.2, 90.0°), 725 oy | 1.8 piksel |
| (740, 0°) | (742.8, 0.3°), 484 oy | 4.8 piksel |
| (410.1, 45°) | (409.3, 44.9°), 663 oy | 1.1 piksel |
| (670.8, 63.4°) | (671.2, 63.3°), 365 oy | 1.0 piksel |

**Yorum.**

- GPU oylama matrisi CPU ile hücre hücre aynıdır (farklı hücre sayısı 0); yani paralel oylama sonucu değiştirmedi. Oylama tam sayı toplamadan ibaret olduğu için oyların eklenme sırası sonucu etkilemez, atomik toplama da oyları kaybettirmez.
- GPU çekirdeği CPU'nun tek thread'li sürümünden yaklaşık 41-45 kat hızlıdır; verinin GPU'ya yüklenip matrisin geri indirilmesi dahil 28 kat. Aktarım maliyeti yaklaşık 0.53 ms'dir (1.691 - 1.158): yalnızca kenar piksel koordinatları (yaklaşık 261 KB) gider ve oylama matrisi (1.44 MB) geri gelir.
- Colab'ın 2 vCPU'su, ikinci thread'le CPU oylamasını yalnızca 1.23 kat hızlandırdı (47.8 ms'den 39.0 ms'ye).
- İki GPU çekirdeği arasındaki fark çok küçüktür (naif 1.066 ms, paylaşılan bellek 1.158 ms; fark 0.09 ms). Paylaşılan belleğin daha hızlı çıkması beklenebilirdi, ama bu problem boyutunda (5.9 milyon oy) iki çekirdek de yaklaşık 1 ms'de biter ve bu fark tek bir ölçümde anlamlı olmayabilir. Olası bir açıklama: paylaşılan bellek çekirdeği theta başına bir blok kullandığı için yalnızca 180 blok açar (T4'ün işlemci sayısına göre az) ve çok sayıda pikselin aynı rho hücresine oy vermesi paylaşılan bellekte de sıralanma yaratır. Bu doğrulanmadı.
- Bu problem GPU için küçüktür; CPU 1 thread'in bile süresi 48 ms'dir. GPU kazancı problem büyüdükçe (örneğin daha ince açı adımıyla daha çok oy) artabilir; bu rapor için denenmedi.

## 7. Sınırlılıklar

- Tüm denemeler tek bir sentetik sahnede yapıldı; eşik değerleri aynı sahnedeki sonuçlara bakılarak seçildi. Farklı görüntülerde yeniden ayarlanmaları gerekir.
- OpenCV Hough fonksiyonlarının parametreleri (özellikle `HoughCircles` param2) ayarlanmadı; sahte çember sayıları bu yüzden yüksek.
- Sahnedeki şekiller 3 piksel kalınlığında çizildi ve Canny her çizgi ya da halka için iki kenar verdiği için, bulunan konum ve yarıçaplarda yaklaşık yarım çizgi kalınlığı kadar (1-3 piksel) bir sapma olması doğaldır.
- Süreler tek ölçümden alındı (5 oylama için ms düzeyinde), tekrar edilmedi; bilgisayarlar arası karşılaştırma yapılmadı.
- Madde 5 için tasarım, hocaya sorulmadan kendi yorumumuzla yapıldı.
- CUDA denemesi tek sahne, tek GPU (Tesla T4) ve Colab'ın 2 vCPU'lu makinesi üzerinde yapıldı. GPU süresi yalnızca oylamayı kapsar; Canny, median ve tepe bulma CPU'da yapıldı ve ölçülmedi. Colab'daki CPU süresi (47.8 ms), Madde 2'deki yerel bilgisayar süresiyle (22.0 ms) karşılaştırılamaz: makine ve döngü sırası farklıdır. İki GPU çekirdeği arasındaki fark tek bir ölçüm kümesine dayanır.
