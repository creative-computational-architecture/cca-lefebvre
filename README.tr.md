<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/media/lefebvre-banner-dark.svg">
  <img src="docs/media/lefebvre-banner-light.svg" width="100%" alt="CCA Lefebvre - nokta bulutlari, mimari rolove ve CAD kaniti">
</picture>

<a href="LICENSE"><img align="right" src="docs/media/license-gpl3-mini.svg" width="116" alt="Lisans: GPL-3.0-or-later"></a>
<img align="right" src="docs/media/cca-mini.svg" width="120" alt="Creative Computational Architecture">

*[English](README.md) | Turkce*

[![Kod: GPL-3.0-or-later](https://img.shields.io/badge/Kod-GPL--3.0--or--later-666666?style=flat-square&labelColor=1A1A1A)](LICENSE) [![Yerel masaustu](https://img.shields.io/badge/Platform-Windows_masaustu-C8102E?style=flat-square&labelColor=1A1A1A)](#deneyin) [![Python ve Qt](https://img.shields.io/badge/Araclar-Python_%2B_Qt_%2B_VTK-666666?style=flat-square&labelColor=1A1A1A)](#mimari) [![Rolove taslagi](https://img.shields.io/badge/Durum-Rolove_taslagi-666666?style=flat-square&labelColor=1A1A1A)](docs/METHOD.md)

**Mimari rolove icin yerel bir nokta bulutu calisma ortami: taramayi inceleyin, geometriyi degerlendirin ve kaniti CAD'e tasiyin.** Lefebvre; tarama yerlestirmeyi, yukseklik ve izdususum gorunumlerini, plan adaylarini, rolove paftalarini ve cizim ciktilarini tek ortamda birlestirir. Ozgun tarama, ondan turetilen yorumdan ayri tutulur.

> **Kod [GPL-3.0-or-later](LICENSE) ile acik kaynaktir.** Asagidaki ornek ve gorseller uygulamayi ve hesaplanan ciktilarini gosterir. Rolove sonuclari incelenip bagimsiz olarak kontrol edilene kadar **SURVEY DRAFT / UNCALIBRATED** durumundadir.

![Studyonun kendi ofisi, yerel Lefebvre calisma ortaminda](docs/media/09-office-native-workspace.png)

**Taradiginiz mekandan baslayin.** Taramalarda mobilyalar, kesintili duvar kaniti ve esit olmayan veri kapsami bulunur. Lefebvre bu kosullari gorunur kilar ve operatorun noktalardan mimari cizimlere ilerlemesini saglar. Donusumun, varsayimlarin ve hata durumlarinin incelenip iyilestirilebilmesi icin kaynak kodunu aciyoruz. Sorun bildirimleri, tekrar uretilebilir ornekler ve katkilar bekliyoruz.

> **Yazarlar / Baglam** - Creative Computational Architecture - Caglar Celik Architects (CCA); Codex, yayin hazirligi ve incelemesine destek verdi. *"Redefining space through computation"* - analiz, matematik, sanat, geometri, felsefe, estetik, mimarlik ve teknoloji arasinda calisan bir tasarim praksisi studyosu.
>
> [Instagram @caglarcelikarchitects](https://www.instagram.com/caglarcelikarchitects) | [caglarcelik.works](https://caglarcelik.works)

---

## Ofis taramasindan cizimlere

Ana uygulama ornegi studyonun kendi ofisinde sinirlanmis bir ust mekan bolgesini kullanir. Kesilip acilmis gorunum taranan geometriyi ortaya cikarir; sinirlar tum ofisin planini degil, secilen bolgeyi tanimlar. Kontrol edilen gorseller yerel calisma ortamini ve turetilen geometrik kaniti gosterir. Ozgun tarama, cekim fotograflari, proje dosyalari ve gercek CAD ciktisi bu deponun disinda tutulur.

| Analiz | CAD onizlemesi |
|---|---|
| ![Ofis rolove kaniti ve aday geometrilerin incelenmesi](docs/media/10-office-survey-analysis.png) | ![Ofisin CAD ciktisinin kontrol edilmis cizim onizlemesi](docs/media/11-office-cad-drawing.png) |
| **Destegi okuyun.** Nokta dagilimini ve rolove hattinin onerdigi geometriyi inceleyin. Destek etiketleri her adayin operator tarafindan incelenmesini destekler. | **Cizimi inceleyin.** Onizleme gercek cikti dosyasindaki nokta izdususumlerini ve duvar adayi katmanlarini gosterir. Kesilmis ic mekan bolgesi yapi adayi veya turetilmis olcu uretmez; gorunus, kesit ve olcu ornekleri ayri matematiksel deneyde bulunur. Ofisin gercek CAD dosyasi dagitilmaz. |

Ice aktarilan veri, 4.788.397 sonlu nokta iceren islenmis bir XYZ PLY dosyasidir; cihazin cekim verisi ice aktarmadan once yerelde donusturuldu. Kamuya acik uygulama cihazin LiData bicimini dogrudan cozmez. Gosterilen rolove `source_axis="CAD_Z_up"` ve acikca secilen 1.0 m en kisa duvar ayarini kullanir. **93 duvar adayi: 4 HIGH, 20 MEDIUM ve 69 LOW** uretir; mobilyalar ve kesintili yuzeyler de operator incelemesi gerektirir. Ozel tutulan DXF 5.406 eleman icerir ve tekrar acildiginda denetim hatasi vermez. Kesilmis bu ic mekan icin yapi, olcu veya kesit elemani turetilmedi.

Ofis uygulamasinin bagimsiz bir kontrol rolovesi bulunmaz ve sayisal dogruluk iddiasi sunmaz. Asagidaki matematiksel deney, girdisi bilinen ayri bir calismadir.

## Deneyin

Calisan bir OpenGL grafik surucusu bulunan 64-bit Windows'ta Python 3.12 kullanin. Uygulama bos bir calisma ortamiyla acilir.

```powershell
git clone https://github.com/creative-computational-architecture/cca-lefebvre.git
cd cca-lefebvre
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\python.exe -m viewer.desktop_app
```

Gercek bir tarama kullanmadan ilk deneme icin matematiksel rolove ornegini uretip acin:

```powershell
.\.venv\Scripts\python.exe examples/synthetic_survey.py --output generated/survey-demo
.\.venv\Scripts\python.exe -m viewer.desktop_app --project generated/survey-demo/input/demo.ccapoints.json --calculate-points --frame-scans
```

Uygulamada taramayi secin. Otomatik rolove paketi icin **Cikti > Otomatik Rolove**, izdususum akisi icin **Cikti > Yuzeyden Plan + Gorunus** yolunu acin. Uretici, [METHOD.md](docs/METHOD.md) icinde belgelenen deney ayarlariyla ayni rolove servisini zaten calistirmistir. Yerel arayuzde Turkce ASCII etiketler kullanilir.

Kurulumla gelen komut `cca-lefebvre` adindadir. Denenen bagimlilik surumleri, yerel pencere dogrulamasi ve paket derlemesi icin [BUILD.md](BUILD.md) dosyasina bakin.

## Neler yapabilirsiniz

- **Taramayi inceleyin.** Proje acin, nokta bulutu kanitini iceri alin, taramalari secin, gorunurlugu yonetin; perspektif, ust, on, yukseklik ve guven gorunumlerini karsilastirin.
- **Taramalari bilincli yerlestirin.** Donusumleri ve hizalama adaylarini inceleyin, olcegi acik tutun, secilen yerlestirmeyi uygulayin veya geri alin.
- **Plani gelistirin.** Izdususum akisini kullanarak dilimleri ve onerilen mimari elemanlari duvar veya aciklik olarak yorumlamadan once inceleyin.
- **Rolove paketi olusturun.** Rolove koordinat cercevesini kestirin, desteklenen duvar parcalarini ve yapi adaylarini turetin; plan, gorunus, kesit ve dilim paftalarini olusturun.
- **Cizim kanitini cikarin.** Rolove paketi DXF katmanlari, olculer ve sayisal aday dosyalari yazar. Dusuk destekli geometri ayirt edilebilir kalir.
- **Kaynagi esas alin.** PLY ozelliklerini koruyan donusumler turetilmis dosyalara yazar. Sentetik deney, rolove oncesi ve sonrasinda girdi hash'ini de kontrol eder.

Genel olcum komutlari, bagimsiz DXF menu ogeleri ve tekil ICP menu ogeleri tamamlanmis is akislari olarak sunulmaz. ReCap/Revit birlikte calismasi burada anlatilan dogrulamanin disindadir.

## Yerel rolove is akisi - sentetik girdi

Bu kareler, uretilen iki bolmeli mekanin gercek masaustu uygulamasindaki ekran goruntuleridir. Calisan kontrolleri, izdususum panelini ve aday listesini gosterir.

| Inceleyin ve gosterin | Cizimi gelistirin |
|---|---|
| ![Yerel calisma ortaminda 176.180 sentetik nokta](docs/media/02-lefebvre-native-workspace.png) | ![Secilmis plan duzlemi ve PNG arti DXF cikti kontroluyle yerel izdususum paneli](docs/media/06-native-survey-projection.png) |
| **Nokta kaniti.** Calisma ortami, tarama gorunurlugu, secim ve kamera kontrolleriyle 176.180 noktayi gosterir. | **Plan ve gorunus.** Bir izdususum yuzeyi tanimlayin, duzlemi inceleyin ve panelin PNG + DXF cikti yolunu kullanin. |
| ![Sentetik mekanin yerel yukseklik-renk gorunumu](docs/media/05-native-height-view.png) | ![Duvar, kapi ve pencere adaylariyla yerel plan-analiz paneli](docs/media/07-native-plan-analysis.png) |
| **Yukseklik gorunumu.** Mekanin dusey dagilimini dogrudan gorunumden okuyun. Renk gosterimi kaynak geometrisini korur. | **Adaylari inceleyin.** Bu etkilesimli deneme 14 eleman bildirir: 10 duvar, 2 kapi ve 2 pencere adayi. Durumlari ADAY olarak gosterilir. |

Etkilesimli plan bulucusu, asagidaki otomatik rolove paketinin 10 duvar parcasindan ayridir. Aciklik adaylari dogrulanmis degildir. Izdususum sonucunu incelemek icin [buyutulmus yerel plan onizlemesini](docs/media/08-native-plan-preview.png) acin.

## Ayri bir matematiksel deney

Deney, ofisin taramasi yerine uretilmis iki bolmeli bir mekandir. Ic boyutlar 8.00 x 5.00 m, duvar kalinligi 0.18 m, sacak 3.00 m ve mahya 3.80 m'dir. Uretici 176.180 noktaya 17 derecelik kaynak donusu ve deterministik Gauss gurultusu ekler. Fotograf, musteri projesi veya dis veri kumesi okumaz.

| Hesaplanan rolove kaniti | Cizim kaniti |
|---|---|
| ![Sentetik duvar destek etiketleri ve dis zarf farklari](docs/media/03-survey-analysis.png) | ![Gercek DXF'ten cizdirilen sentetik plan, gorunus, kesit ve CAD olculeri](docs/media/04-cad-drawing-sheet.png) |
| **10 duvar parcasi:** 7 HIGH, 1 MEDIUM ve 2 LOW destek etiketi. Iki dusuk destekli parca elle inceleme icin gorunur kalir. | **1.832 DXF entity:** 9 olcu, santimetre birimi ve yeniden acilista sifir audit hatasi. Grafik, uretilen DXF'ten cizdirilir. |

| Kontrol | Gozlenen sonuc |
|---|---:|
| Uretilen / islenen nokta | 176.180 / 176.180 |
| Ham girdinin onceki / sonraki SHA-256 degeri | Ayni |
| Rijit cercevenin olcegi / determinanti | 1 / +1 |
| Duvar / enine duvar / bolme konumu / yapi adayi | 10 / 3 / 3 / 2 |
| Zemin duzlemi uyum sigma degeri | 1.23722 mm |
| Eklenen gurultunun XYZ RMS degeri | 2.59838 mm |
| Dis zarf farklari X / Y / Z | 12.167 / 11.415 / 10.225 mm |
| DXF birimi (`$INSUNITS`) / audit hatasi | cm (5) / 0 |

**Uyum sigma degeri ve zarf farklari gercek dunyadaki rolove hassasiyeti degildir.** Cikarilan bolme genislikleri, nominal 4.00 m merkez cizgisi acikliklarina karsilik 3.91 ve 4.09 m'dir. Cikarilan derinlik 5.00 m yerine 5.20 m, sacak ise 3.00 m yerine 3.09 m'dir. HIGH destek etiketi bu sapmalarla birlikte bulunabilir.

Gercek [sentetik DXF'i](docs/demo/synthetic-survey.dxf), [sayisal ozeti](docs/demo/summary.json) ve [tekrar uretim notlarini](docs/demo/README.md) acin. Ureticideki kapi ve pencere konumlari bilinen aciklamalardir; bu deney aciklik bulmayi dogrulamaz. Etkilesimli plan analizi ayri bir is akisidir ve farkli sayida aday uretebilir.

## Yontemin ici

```text
acik girdi + eksen / yerlestirme
    -> alcak noktalarla zemin RANSAC + SVD duzlem iyilestirmesi
    -> metrik rijit cerceve + yatay yon aramasi
    -> yukseklik bantlari + doluluk / izdususum nokta kaniti
    -> destekli duvar parcalari + guven / yapi adaylari
    -> plan + gorunus + kesit + olculer / DXF katmanlari
    -> audit + kaynak hash'i + operator incelemesi
```

| Asama | Kodun yaptigi |
|---|---|
| Cerceve | Zemin duzlemini kestirir, taramayi yataylar, yatay yonleri arar ve olcek 1 olan uygun rijit donusumu kaydeder |
| Dilimler | Belirlenmis yukseklik bantlarinda doluluk rasterlari ve cati yuksekligi izdususumu uretir |
| Duvarlar | Histogram tepelerini ve desteklenen parcalari bulur, uyumlu parcalari birlestirir ve yukseklik bantlarindaki destegi etiketler |
| Yorum | Hizalanmis kanittan enine duvar, bolme ve yapi adaylarini turetir |
| Cikti | Paftalari, sayisal adaylari ve kaynak izdususumleri, geometri, dusuk destekli duvarlar, kesit, gorunus ve olculer icin ayri DXF katmanlarini yazar |
| Inceleme | Zayif kaniti korur ve varsayimlari bildirir; yorumu operator kontrol etmelidir |

Varsayilan parametreler, deneyde degistirilen ayarlar ve gozlenen sapmalar icin [METHOD.md](docs/METHOD.md) dosyasina bakin. [ARTICLE.md](docs/ARTICLE.md), cizim onerilerinin neden kendi kanitina bagli tutuldugunu anlatir.

## Mimari

```text
viewer/      Qt masaustu, proje modeli, tarama yerlestirme ve is akisi servisleri
geometry/    rijit donusumler, nokta ozellikleri, rolove cercevesi ve adaylar
export/      rolove paketi, paftalar, DXF ve turetilmis cizim ciktilari
ingest/      genel nokta bulutu ve CAD okuyuculari
field/       cekim derinligi ve renk donusumu yardimcilari
examples/    matematiksel ureticiler ve tekrar uretilebilir rolove ornegi
tests/       kaynak koruma ve sentetik regresyon kontrolleri
docs/        kontrol edilmis medya, yontem notlari ve sentetik CAD kaniti
```

Bu, PySide6, PyVista/VTK ve Open3D kullanan bagimsiz bir Python uygulamasidir. CloudCompare ayri bir arac ve is akisi referansidir. Bu depo CloudCompare kaynak kodu, ikon veya binary icermez; CloudCompare kaynak fork'u degildir. Dis bagimliliklar kendi lisanslarini korur.

Ana calisma ortami model servisi olmadan calisir. Istege bagli yerel planlayici, acikca yapilandirilmis bir endpoint ister. Istege bagli standart girdi/cikti kontrol koprusu `.[control]` ile kurulur; yapilandirma ve kapsam [BUILD.md](BUILD.md) icinde anlatilir.

## Dosyalar ve tekrar uretim

| Dosya | Amac |
|---|---|
| [examples/synthetic_survey.py](examples/synthetic_survey.py) | Kurgusal girdiyi uretir ve uygulamanin gercek rolove servisini cagirir |
| [docs/demo/summary.json](docs/demo/summary.json) | Kaynak bilgisi, ayarlar, sayilar, hash'ler, cerceve ve bilinen geometriyle karsilastirma |
| [docs/demo/synthetic-survey.dxf](docs/demo/synthetic-survey.dxf) | Gercek sentetik plan, gorunus, kesit ve olcu entity'leri |
| [docs/METHOD.md](docs/METHOD.md) | Algoritma, parametreler ve sinirlar |
| [BUILD.md](BUILD.md) | Kurulum, yerel pencere kontrolleri ve paket derlemesi |
| [RELEASE_CHECKS.md](RELEASE_CHECKS.md) | Bu yayin icin yapilan dogrulama |

Uretilen nokta bulutlari ve yerel projeler Git tarafindan yok sayilir. Bu acik yayinda yalniz incelenmis kaynak kodu, lisans atiflari, belge gorselleri ve matematiksel olarak turetilmis CAD kaniti bulunur. Ofis girdisi ve gercek CAD dosyalari ozel kalir.

## Yol haritasi - katkilar bekleniyor

- Sentetik regresyon orneklerini donmus mekanlar, eksik kapsama, egri duvarlar ve mobilyalarla genisletmek.
- Olcumleri bagimsiz olculmus kontrol geometrisiyle karsilastirmak.
- Acikliklar dahil aday incelemesini ve guven aciklamalarini iyilestirmek.
- Devre disi olcum ve bagimsiz cikti is akislarini tamamlayip dogrulamak.
- ReCap/Revit aktarimini hassas veri icermeyen kontrollu orneklerle dogrulamak.
- Ingilizce arayuz kapsamiyla kurulum rehberini iyilestirmek.

Kucuk bir matematiksel girdi, beklenen geometri ve tekrar uretilebilir hata iceren katkilar en faydali olanlardir. Kaynak kurallari icin [AGENTS.md](AGENTS.md) dosyasina bakin.

## Araclar ve Atiflar

| Arac | Rol |
|---|---|
| Python / NumPy / SciPy | Sayisal geometri ve analiz |
| PySide6 / Qt | Yerel masaustu arayuzu |
| PyVista / PyVistaQt / VTK | Nokta bulutu ve geometrik gorsellestirme |
| Open3D | Nokta bulutu ve turetilmis yuzey islemleri |
| ezdxf | CAD entity uretimi, okuma ve audit |
| Matplotlib | Hesaplanan ciktidan uretilen rolove paftalari ve grafikler |
| CloudCompare | Bagimsiz nokta bulutu araci ve ilgili is akisi referansi |
| Codex | Acik kaynak hazirligi, belgeleme ve inceleme destegi |

Bilesen lisanslari ve zorunlu atiflar icin [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) ve [licenses/](licenses/) klasorune bakin. Denenen calisma ortami surumleri [requirements-tested.txt](requirements-tested.txt) icindedir.

## Lisans

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/media/cca-mark-dark.svg">
    <img src="docs/media/cca-mark.svg" width="120" alt="Creative Computational Architecture - CCA">
  </picture>
</p>

<p align="center"><a href="LICENSE"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/media/license-gpl3-dark.svg">
  <img src="docs/media/license-gpl3.svg" width="400" alt="GPL-3.0-or-later">
</picture></a></p>

CCA kaynak kodu [GPL-3.0-or-later](LICENSE) ile lisanslanir. Yeniden dagitimda lisansi, karsilik gelen kaynak kodu yukumluluklerini ve ucuncu taraf atiflarini koruyun. Bagimliliklar kendi telif ve lisans kosullarini korur. Kanonik GPL metni [COPYING](COPYING) icindedir.

Copyright (c) 2026 Creative Computational Architecture - Caglar Celik Architects (CCA).
