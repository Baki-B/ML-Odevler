# YZM212 Makine Öğrenmesi - Ödev 2: MLE ile Akıllı Şehir Planlaması

## Problem Tanımı
Bu projede, bir belediyenin en yoğun ana caddesinden geçen araç sayılarının trafik yoğunluğu modeli çıkarılmak istenmiştir. 
Amaç, gelecekteki trafiği tahmin etmek için Poisson Dağılımı varsayımı altında Maximum Likelihood Estimation (MLE) yöntemi kullanılarak 
en uygun trafik parametresini ($\lambda$) bulmaktır.

## Veri Seti
Şehrin en yoğun caddesinden 1 dakikada geçen araç sayılarını gösteren 14 adet gözlemden oluşmaktadır: 
`[12, 15, 10, 8, 14, 11, 13, 16, 9, 12, 11, 14, 10, 15]`

## Yöntem
1. **Analitik MLE:** Poisson dağılımı için Log-Likelihood fonksiyonu türetilmiş ve türevi sıfıra eşitlenerek 
en iyi tahminin (MLE) verilerin aritmetik ortalaması olduğu ispatlanmıştır.
2. **Sayısal MLE:** Python `scipy.optimize` kütüphanesi kullanılarak Negatif Log-Olabilirlik (NLL) fonksiyonu minimize edilmiş ve 
$\lambda \approx 12.14$ değeri bulunmuştur.
3. **Görselleştirme:** Gerçek verinin histogramı ile teorik Poisson PMF grafiği üst üste çizdirilerek modelin uyumu (fit) test edilmiştir.

## Sonuçlar ve Yorum
Grafik analizinde, MLE yöntemiyle hesaplanan teorik Poisson modelinin merkezi ile gerçek trafik verilerinin genel yayılımının 
büyük ölçüde örtüştüğü (12 civarında) görülmüştür. Gözlem sayısının ($n=14$) küçüklüğünden kaynaklanan anlık dalgalanmalara rağmen, 
model gerçek verilere oldukça başarılı bir uyum sağlamıştır. 

Ayrıca "Outlier" analizinde, veri setine yanlışlıkla eklenecek "200" gibi tek bir hatalı değerin ortalamayı ve dolayısıyla modeli tamamen bozacağı saptanmıştır. 
Bunun da belediye planlamasında "gereksiz yol genişletme" gibi maliyetli fiyaskolara yol açabileceği tartışılmış, 
veri temizliğinin (Garbage In, Garbage Out) önemi vurgulanmıştır.
