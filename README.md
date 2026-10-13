# MNIST Veri Seti ile MLP Tabanlı El Yazısı Rakam Sınıflandırma

Bu proje, PyTorch kütüphanesi ve Çok Katmanlı Algılayıcı (**MLP - Multi-Layer Perceptron**) mimarisi kullanılarak MNIST el yazısı rakam veri setinin sınıflandırılmasını, model eğitim optimizasyonunu ve gerçek dünya el yazısı (okul numarası) verileriyle test edilmesini içermektedir.

---

## 1. Proje Mimarisi ve Teknik Parametreler
* **Model Tipi:** Yapay Sinir Ağı (ANN) / MLP (Evrişimli katman içermez, tamamen tam bağlantılı / *fully-connected* lineer katmanlardan oluşur).
* **Girdi Katmanı:** 28x28 = 784 piksel (Flatten edilmiştir).
* **Gizli Katmanlar (Hidden Layers):** 
  * 1. Gizli Katman: `nn.Linear(784, 256)` + ReLU aktivasyon fonksiyonu
  * 2. Gizli Katman: `nn.Linear(256, 128)` + ReLU aktivasyon fonksiyonu
* **Çıkış Katmanı:** `nn.Linear(128, 10)` (0-9 rakam sınıfları için 10 nöron).
* **Hiperparametreler ve Donanım:**
  * **Optimizasyon Algoritması:** Adam (`learning_rate = 0.001`)
  * **Kayıp Fonksiyonu (Loss Function):** CrossEntropyLoss
  * **Epoch Sayısı:** 5
  * **Donanım:** NVIDIA GeForce RTX 4070 Laptop GPU

---

## 2. Veri Seti Bölümleme ve Eğitim Metrikleri
* **Veri Ayrımı:** Orijinal MNIST eğitim setindeki 60.000 veri, `torch.utils.data.random_split` kullanılarak **%85 Eğitim (51.000 örnek)** ve **%15 Doğrulama (9.000 örnek)** olacak şekilde bölünmüştür. Test aşamasında ise standart 10.000'lik test kümesi kullanılmıştır.
* **Normalizasyon:** Görüntü pikselleri `transforms.Normalize((0.5,), (0.5,))` ile ölçeklendirilmiştir.
* **Eğitim Süresi:** RTX 4070 GPU optimizasyonu sayesinde 5 epoch'luk eğitim **88.45 saniye** sürmüştür.

### Epoch İlerleme Özeti:
* **Epoch 1/5:** Train Loss: 0.365 | Train Acc: 89.01% | Val Loss: 0.192 | Val Acc: 93.92%
* **Epoch 2/5:** Train Loss: 0.160 | Train Acc: 95.16% | Val Loss: 0.130 | Val Acc: 95.96%
* **Epoch 3/5:** Train Loss: 0.121 | Train Acc: 96.25% | Val Loss: 0.116 | Val Acc: 96.36%
* **Epoch 4/5:** Train Loss: 0.094 | Train Acc: 97.09% | Val Loss: 0.111 | Val Acc: 96.37%
* **Epoch 5/5:** Train Loss: 0.080 | Train Acc: 97.45% | Val Loss: 0.109 | Val Acc: **96.68%**

---

## 3. Model Performans Değerlendirmesi
Model, eğitim sürecinden hiç geçmemiş 10.000 adet test görüntüsü üzerinde değerlendirilmiş ve nihai olarak **%96.76 Test Accuracy (Genel Başarı)** elde etmiştir.

---

## 4. Gerçek Dünya Testi ve "Domain Gap" (Dağılım Kayması) Analizi
Projenin final aşamasında, modelin genelleştirme yeteneğini ölçmek amacıyla fiziksel kağıda yazılan okul numarasının rakamları tek tek kesilerek (`rakamlar/` klasörü altında) modele test ettirilmiştir.

* **Gözlem:** Model dijital MNIST testlerinde %96+ başarı yakalarken, telefon kamerasıyla çekilen ham el yazısı fotoğraflarında bazı tahmin sapmaları (örneğin bazı rakamların farklı sınıflara Atfedilmesi) göstermiştir.
* **Akademik Yorum:** Bu durum, eğitim veri setindeki sentetik/dijital rakam dağılımı ile gerçek dünya kağıt-kalem verisi arasındaki **piksel kontrastı, aydınlatma ve çizim karakteristiği farkından (Domain Gap / Dağılım Kayması)** kaynaklanmaktadır. Gerçek dünya yapay zeka projelerinde bu durum beklenen ve raporlanması gereken teknik bir olgudur.