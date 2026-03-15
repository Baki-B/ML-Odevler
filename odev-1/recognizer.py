import numpy as np
from hmmlearn import hmm


# Gerçekte ses frekans özelliklerini simüle ediyoruz.
# Veri yapısı: [Zaman Adımı, Öznitelik Sayısı]

# "EV" kelimesi için simüle edilmiş veri
# EV verisi ortalaması 0.0 olan bir dağılımdan gelsin (kısa, 5 zaman adımı)
ornek_sayisi = 15
ev_zaman_adimi = 5
X_ev = np.random.normal(loc=0.0, scale=1.0, size=(ornek_sayisi * ev_zaman_adimi, 2))
lengths_ev = [ev_zaman_adimi] * ornek_sayisi

# "OKUL" kelimesi için simüle edilmiş veri
# OKUL verisi ortalaması 5.0 olan bir dağılımdan gelsin (uzun, 8 zaman adımı)
okul_zaman_adimi = 8
X_okul = np.random.normal(loc=5.0, scale=1.0, size=(ornek_sayisi * okul_zaman_adimi, 2))
lengths_okul = [okul_zaman_adimi] * ornek_sayisi


# EV modeli (2 farklı ses biriminden/durumdan oluştuğunu varsayalım)
model_ev = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
model_ev.fit(X_ev, lengths_ev)

# OKUL modeli (4 farklı ses biriminden/durumdan oluştuğunu varsayalım)
model_okul = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=100, random_state=42)
model_okul.fit(X_okul, lengths_okul)


def kelime_tanima(test_verisi):
    """
    Yeni bir ses verisinin (gözlem dizisi) hangi modele ait olduğunu Log-Likelihood (score) değerlerini karşilaştirarak bulur.
    """
    # Her iki model için Log-Likelihood değerlerini hesapla
    score_ev = model_ev.score(test_verisi)
    score_okul = model_okul.score(test_verisi)
    
    print(f"Log-Likelihood (EV): {score_ev:.2f}")
    print(f"Log-Likelihood (OKUL): {score_okul:.2f}")
    
    # Hangi modelin olasılığı daha yüksekse onu seç (max alma)
    if score_ev > score_okul:
        print("💡 Sonuç: Söylenen kelime büyük ihtimalle 'EV'")
        return "EV"
    else:
        print("💡 Sonuç: Söylenen kelime büyük ihtimalle 'OKUL'")
        return "OKUL"

# TEST
print("--- TEST 1: Yeni bir 'EV' verisi deniyoruz ---")
# EV verisine benzer yeni bir veri üretiyoruz (ortalaması 0'a yakın)
test_ev_gibi = np.random.normal(loc=0.1, scale=1.0, size=(5, 2))
kelime_tanima(test_ev_gibi)

print("\n--- TEST 2: Yeni bir 'OKUL' verisi deniyoruz ---")
# OKUL verisine benzer yeni bir veri üretiyoruz (ortalaması 5'e yakın)
test_okul_gibi = np.random.normal(loc=4.8, scale=1.0, size=(8, 2))
kelime_tanima(test_okul_gibi)