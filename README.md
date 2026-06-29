# yeniBrainMr

Basit bir beyin MR görüntü segmentasyon projesi (eğitim, değerlendirme ve görselleştirme araçları).

## İçerik
- `scripts/` — eğitim, test ve görselleştirme yardımcı betikleri
- `src/brain_mr_seg/` — veri yükleme, model ve metrik tanımları
- `outputs/` — eğitim çıktı dosyaları, modeller ve loglar (bu depo içinde tutulmaz)

## Hızlı Başlangıç
1. Bağımlılıkları kur:

   pip install -r requirements.txt

2. Eğitmek için:

   python scripts/train.py

3. Test / Değerlendirme:

   python scripts/test.py
   python scripts/evaluate_and_visualize.py

## Notlar
- Büyük model ağırlıkları repoda tutulmaz; `outputs/checkpoints/` dizini `.gitignore` ile hariç tutulmuştur. GitHub'a büyük dosyalar eklemek isterseniz `git-lfs` kullanın.
- Repo sahibi: `htcunl`

## Lisans
İstediğiniz lisansı burada belirtebilirsiniz.
