# Segmentasi Citra — Tugas 3 IF4073

Aplikasi MATLAB untuk segmentasi citra dengan tiga pendekatan, disatukan dalam satu GUI (`SegmentasiGUI.m`):

- **B. Deteksi tepi** — operator Laplace, LoG, Sobel, Prewitt, dan Roberts diimplementasikan sendiri; Canny memakai `edge` bawaan MATLAB. Tepi ditutup secara morfologis, lubangnya diisi, lalu region terbesar diambil sebagai objek.
- **C. Split and merge** — quadtree split dengan predikat simpangan baku, lalu region bertetangga digabung bila jarak rata-rata warnanya di bawah threshold.
- **D. Segmentasi semantik** — DeepLabV3 + MobileNetV3-Large (pretrained COCO/VOC dari torchvision) di-fine-tune pada CamVid dengan 11 kelas. Bagian ini ditulis dalam Python dan dipanggil dari GUI MATLAB.

## Dependensi

- MATLAB R2020b atau lebih baru (`uifigure`, `uigridlayout`)
- Image Processing Toolbox, hanya untuk operator Canny
- Python 3.10+ dengan paket di `semantic/requirements.txt` (PyTorch, torchvision, NumPy, Pillow)

## Menjalankan

Siapkan lingkungan Python di root repositori. GUI otomatis memakai `.venv` ini:

```bash
python3 -m venv .venv
.venv/bin/pip install -r semantic/requirements.txt
```

Di Windows, pakai `.venv\Scripts\pip`.

Lalu buka MATLAB di root repositori dan jalankan:

```matlab
SegmentasiGUI
```

Klik **Buka citra**, pilih tab, atur parameter, lalu klik **Proses**. Citra uji BSDS500 diletakkan di `bsds500_test_8/`.

## Segmentasi semantik (bagian D)

### Dataset CamVid

Sumber resmi: [701_StillsRaw_full.zip](http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip) dan [LabeledApproved_full.zip](http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/LabeledApproved_full.zip). Kalau server UCL tidak dapat diakses, gunakan mirror fast.ai [camvid.tgz](https://s3.amazonaws.com/fast-ai-imagelocal/camvid.tgz). Isinya 701 citra yang sama, dengan label 32 kelas dalam bentuk indeks.

Letakkan kedua zip UCL **atau** `camvid.tgz` di `data/camvid/`, lalu jalankan:

```bash
cd semantic
../.venv/bin/python camvid.py
```

Script ini memetakan 32 kelas asli ke 11 kelas (Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist). Kelas Void diabaikan saat training. Pemetaannya ada di `CLASS_GROUPS` pada `semantic/camvid.py`. Data dibagi acak 60/20/20 dengan seed tetap ke `data/camvid/{train,val,test}/{images,labels}`.

### Training dan evaluasi

```bash
../.venv/bin/python train.py
```

- Citra di-resize ke 360×480, dengan horizontal flip acak sebagai augmentasi.
- Loss memakai cross-entropy berbobot median frequency balancing, dioptimasi dengan AdamW (lr 1e-4) dan cosine schedule selama 30 epoch.
- Bobot dengan mIoU validasi terbaik disimpan di `semantic/weights/deeplabv3_mobilenet_camvid.pt`.
- Setelah training selesai, model dievaluasi pada split test. Hasilnya (global accuracy, mean accuracy, mIoU, IoU per kelas) dicetak dan disimpan di `semantic/weights/test_metrics.json`.

GPU dipakai otomatis bila tersedia. Di CPU 8 core, satu epoch butuh sekitar 3 menit. Evaluasi ulang tanpa training:

```bash
../.venv/bin/python evaluate.py
```

### Bobot model

Bobot hasil fine-tuning tidak disimpan di repositori. Unduh `deeplabv3_mobilenet_camvid.pt` dari [TAUTAN_BOBOT] dan letakkan di `semantic/weights/`. Tab D baru bisa dipakai setelah file ini ada, atau setelah `train.py` dijalankan sendiri.

### Inferensi dari GUI

Tab **D. Segmentasi Semantik** memanggil `semantic/predict.py` lewat `SemanticSegment.m`. Tab ini menampilkan citra masukan, peta kelas, overlay, dan legenda warna 11 kelas. Citra dari `data/camvid/test/images` juga ditampilkan bersama ground truth dan akurasi pikselnya.

Dari terminal:

```bash
../.venv/bin/python predict.py citra.png peta_label.png
```

`peta_label.png` berisi indeks kelas 0–10 sesuai urutan di atas.
