# Uzak Bir Galaksinin Parlaklık Analizi

## Problem Tanımı
Gürültülü gözlem verisinden bir gök cisminin gerçek parlaklığını ve ölçüm belirsizliğini
Bayesyen çıkarım yöntemiyle tahmin etmek.

## Veri
Sentetik gözlem verisi. Gerçek parlaklık 150.0, gözlem gürültüsü 10.0, gözlem sayısı 50.
Tekrarlanabilirlik için np.random.seed(42) kullanılmıştır.

## Yöntem
MCMC tabanlı Bayesyen çıkarım (emcee kütüphanesi). 32 walker, 2000 adım, 500 adım burn-in.
Dört farklı deney yapılmıştır: farklı başlangıç noktaları, dar prior ve az veri etkileri incelenmiştir.

## Sonuçlar
Ana deney (geniş prior, n=50) için µ median ≈ 147.79, σ median ≈ 9.49, hata payı ~%1.47.

## Yorum
Yanlış prior posterior dağılımını tamamen bozar. Az veri belirsizliği artırır ancak doğruluğu
şansa bırakır. µ ve σ parametreleri birbirinden bağımsızdır.