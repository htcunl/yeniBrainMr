
# yeniBrainMr

Bu proje, beyin MR görüntüleri üzerinde segmentasyon görevleri için bir araştırma/deney hattı sağlar. Eğitim, değerlendirme ve sonuçların görselleştirilmesi için betikler ve yeniden üretilebilir çalışma dizini düzeni içerir.

## Proje Özeti
- Amaç: Beyin MR görüntülerinde belirli yapıları (ör. lezyon, doku sınıfları) otomatik olarak segment etmek.
- Kullanılan araçlar: TensorFlow/Keras tabanlı modeller, özel kayıp/metric fonksiyonları ve eğitim/izleme betikleri.

## Veri
- Girdi: Dönüştürülmüş MR görüntü setleri (niyetlenen eğitim/validation/test split'leri `splits.json` ve `splits_80_10_10_locked_test.json` içinde saklı).
- Not: Orijinal ham veri bu repoda yer almaz; veri ön işleme sizin tarafınızdan yapılmalıdır.

## Model ve Eğitim
- Model: `src/brain_mr_seg/model.py` içinde tanımlı (özelleştirilebilir mimari ve hiperparametreler).
- Kayıp / metri̇kler: `src/brain_mr_seg/losses.py` ve `src/brain_mr_seg/metrics.py`.
- Eğitim: `scripts/train.py` betiği ile başlatılır. Eğitim konfigürasyonu `outputs/training_run_config.json` içinde veya betik argümanlarıyla verilebilir.

## Değerlendirme ve Görselleştirme
- Test çalıştırmaları için `scripts/test.py` kullanılabilir.
- Sonuçların analizi ve görselleştirmesi `scripts/evaluate_and_visualize.py` ve `scripts/plot_comparison_metrics_bar.py` ile yapılır.

## Klasör Yapısı (Önemli)
- `scripts/` — Çalıştırılabilir yardımcı betikler
- `src/brain_mr_seg/` — Veri, model ve metrik kodu
- `outputs/` — Eğitim sırasında üretilen modeller, loglar ve görseller (bu repoda tutulmaz)

## Hızlı Başlangıç
1. Bağımlılıkları kur:

```powershell
pip install -r requirements.txt
```

2. Eğitim (örnek):

```powershell
python scripts/train.py --config outputs/training_run_config.json
```

3. Test ve değerlendirme:

```powershell
python scripts/test.py
python scripts/evaluate_and_visualize.py
```

## Büyük Ağırlıklar ve Depo Politikası
- Model ağırlıkları (`*.h5`, checkpoint'ler) repoda saklanmaz; `.gitignore` ile dışlanmıştır.
- Eğer geçmişten tamamen kaldırmak isterseniz `git filter-repo` veya BFG ile yardım edebilirim.

## Katkı & İletişim
- Repo sahibi: `htcunl` — değişiklik yapmak isterseniz fork ve pull request açabilirsiniz.

## Lisans
- Bu projeye uygun bir lisans eklemek için bir `LICENSE` dosyası oluşturabilirsiniz.

