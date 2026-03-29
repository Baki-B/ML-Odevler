**1. Soru: Makine Öğrenmesi, Matris Manipülasyonu, Özdeğerler ve Özvektörlerin İlişkisi ve Kullanım Alanları**
- **Matris Manipülasyonu (Matrix Manipulation):** Matrisler, sayıların veya ifadelerin dikdörtgen bir tablo halinde düzenlenmiş halleridir. Matris manipülasyonu; matrisler üzerinde toplama, çıkarma, skalerle çarpma, matris çarpımı, devriğini alma (transpose) ve tersini alma (inverse) gibi doğrusal cebir işlemlerini ifade eder.
- **Özdeğerler (Eigenvalues) ve Özvektörler (Eigenvectors):** Kare bir $A$ matrisi için, $A \cdot x = \lambda \cdot x$ denklemini sağlayan sıfırdan farklı bir $x$ vektörüne "özvektör", bu denklemi sağlayan $\lambda$ skaler sayısına ise "özdeğer" denir. Basit bir ifadeyle, bir matris (doğrusal dönüşüm) bir özvektöre uygulandığında, vektörün yönü değişmez; sadece boyutu $\lambda$ (özdeğer) kadar uzar, kısalır veya yönü tersine döner.

**Makine Öğrenmesi ile İlişkisi**
Makine öğrenmesinde veriler ve algoritmaların parametreleri matrisler ve vektörler şeklinde temsil edilir. Bir veri setindeki her satır bir veri noktasını, her sütun ise bir özelliği (feature) temsil eden bir matris oluşturur.
1. **Matris Manipülasyonunun Rolü:** Model eğitimi, veri setindeki örüntüleri öğrenmek için sürekli olarak matrislerin birbirleriyle çarpılması ve güncellenmesi sürecidir. İleri yayılım (forward propagation) ve geri yayılım (backpropagation) gibi temel hesaplamalar, işlem gücünü verimli kullanmak adına matris manipülasyonları ile vektörize edilir.
2. **Özdeğerler ve Özvektörlerin Rolü:** Özdeğerler ve özvektörler, makine öğrenmesinde genellikle veri matrislerinin "içsel yapısını" veya "kök bilgisini" anlamak için kullanılır. Özellikle verinin yayılımı (varyans) ve veriyi temsil eden temel eksenlerin bulunmasında kritik bir rol oynarlar. Özvektörler yeni veri uzayının yönlerini belirlerken, özdeğerler o yöndeki bilgi miktarının (varyansın) büyüklüğünü temsil eder.

**Kullanıldığı Yöntemler ve Yaklaşımlar**
Matris Manipülasyonu ile Özdeğerler-Özvektörler, makine öğrenmesi ve veri biliminde başlıca şu yöntemlerde kullanılmaktadır:
1. **Temel Bileşenler Analizi (PCA - Principal Component Analysis):** Veri setinin boyutunu (özellik sayısını) küçültürken, bilgi kaybını (varyansı) en aza indirmeyi amaçlayan bir tekniktir. Bu işlem yapılırken, verinin kovaryans matrisinin **özdeğerleri ve özvektörleri** hesaplanır. En yüksek özdeğere sahip özvektörler, verideki en önemli temel bileşenlerdir.
2. **Tekil Değer Ayrışımı (SVD - Singular Value Decomposition):** Özellikle doğal dil işleme (NLP), tavsiye sistemleri (recommender systems - örneğin Netflix film önerileri) ve görüntü sıkıştırma alanlarında kullanılır. Karmaşık matrisleri, manipülasyonu kolay daha basit alt matrislere ve vektörlere ayırarak hesaplama verimliliği sağlar.
3. **Yapay Sinir Ağları ve Derin Öğrenme (Neural Networks):** Derin öğrenme mimarilerindeki ağırlıklar (weights) matris olarak tutulur. Aktivasyonların hesaplanması ve maliyet fonksiyonunun optimize edilmesi süreci tamamen ileri seviye **matris manipülasyonlarına** (nokta çarpımlar vs.) dayanır.
4. **Spektral Kümeleme (Spectral Clustering) ve Eigenfaces:** Görüntü işlemede (özellikle yüz tanıma sistemlerinde - Eigenfaces), yüzleri temel bileşenlerine ayırmak için doğrudan özvektörler kullanılır. Spektral kümeleme ise veri noktaları arasındaki benzerlik matrisinin özdeğerlerini kullanarak veriyi gruplara ayırır.

**2. Soru: "numpy.linalg.eig" Fonksiyonunun İncelenmesi**
NumPy'ın dokümantasyonunu ve GitHub'daki kaynak kodlarını incelediğimde numpy.linalg.eig fonksiyonunun çalışma mantığıyla ilgili şu sonuçlara ulaştım:
Öncelikle dokümantasyona baktığımızda, bu fonksiyon içine parametre olarak karesel bir matris alıyor. Eğer verdiğimiz matris karesel değilse doğrudan hata (LinAlgError) fırlatıyor. İşlem sonucunda ise bize iki tane çıktı veriyor: w ve v.
- w dizisi matrisin özdeğerlerini veriyor. Dokümantasyonda özellikle dikkatimi çeken kısım, bu özdeğerlerin belirli bir sırayla (örneğin büyükten küçüğe) döndürülmesinin garanti edilmemesi.
- v ise özvektörleri barındıran karesel matris. Burada kafa karıştırabilecek ve kod yazarken dikkat edilmesi gereken en önemli nokta, özvektörlerin satırlarda değil **sütunlarda** yer alması. Yani w[0] özdeğerinin özvektörünü almak istersek v[:, 0] diyerek ilgili sütunu çekmemiz gerekiyor.
- Ayrıca dokümantasyonda küçük bir ipucu daha var: Eğer elimizdeki matris simetrik bir matrisse, eig yerine linalg.eigh fonksiyonunu kullanmamız tavsiye ediliyor. Çünkü eigh, simetrik matrisler için çok daha hızlı ve kararlı sonuçlar veriyormuş.

İşin arka planına, yani GitHub'daki kaynak kodlarına (numpy/linalg klasörü) indiğimde ise hesaplamaların aslında sıfırdan saf Python ile yapılmadığını fark ettim. Python tarafındaki kod daha çok bir "düzenleyici" gibi çalışıyor. Biz matrisi fonksiyona verdiğimizde, kod önce veri tiplerini kontrol ediyor. Örneğin matris elemanlarını tam sayı olarak girdiysek, özdeğerler genellikle küsuratlı çıkacağı için NumPy bunu arka planda otomatik olarak ondalıklı veya karmaşık sayı tiplerine dönüştürüyor.
Asıl ağır matematiksel hesaplama ise alt seviyede, bilimsel hesaplamalarda endüstri standardı olan **LAPACK** kütüphanesine yaptırılıyor. Kaynak kodda, verinin tipine göre geev isimli LAPACK rutinlerinin çağrıldığını gördüm. Kısacası sürecin özeti şu: Biz eig() fonksiyonunu çağırıyoruz, NumPy girdiğimiz veriyi kontrol edip C veya Fortran'ın anlayacağı alt seviye bir formata sokuyor, iş yükünü LAPACK kütüphanesine yaptırıyor ve oradan dönen ham sonuçları tekrar alıştığımız NumPy array'lerine çevirip ekranımıza basıyor.

**3. Soru: Repository Referansıyla Manuel Hesaplama ve NumPy Karşılaştırması**
Belirtilen GitHub reposu (LucasBN), NumPy'ın hazır eig fonksiyonunu kullanmak yerine, temel doğrusal cebir kurallarını koda dökerek karakteristik denklem ($\det(A - \lambda I) = 0$) üzerinden manuel bir çözüm yöntemi sunmaktadır.
Bu mantığı referans alarak $2 \times 2$ boyutunda bir matris üzerinde manuel hesaplama kodunu ve hemen ardından NumPy'ın eig fonksiyonu ile çözümünü uyguladım.

Sonuçların Karşılaştırılması ve Yorumlanması
Kodu çalıştırdığımızda elde edilen sonuçları şu şekilde yorumlayabiliriz:
Özdeğerler Birebir Aynı: Her iki yöntem de özdeğerleri $5$ ve $2$ olarak bulmaktadır. Karakteristik denklemi köklerine ayırmak ($(\lambda - 5)(\lambda - 2) = 0$), küçük boyutlu matrislerde tamamen doğru ve tutarlı çalışmaktadır.
Özvektörlerin Yönü ve İşareti: İki çıktıyı karşılaştırdığımızda, manuel hesaplanan özvektör ile NumPy özvektörünün sayısal değerlerinin aynı olduğunu, ancak işaretlerinin (artı/eksi) ters olabildiğini görürüz. Örneğin; biri $\begin{bmatrix} 0.707 \\ 0.707 \end{bmatrix}$ iken diğeri $\begin{bmatrix} -0.707 \\ -0.707 \end{bmatrix}$ çıkabilir. Doğrusal cebirde her iki sonuç da matematiksel olarak doğrudur; çünkü özvektörler bir noktayı değil bir "doğrultuyu" temsil eder. Vektörün kendisi ile $-1$ ile çarpılmış hali aynı ekseni ifade etmektedir.
Performans Farkı: Manuel kodumuz $2 \times 2$ bir matriste sorunsuz çalışsa da, matris boyutu büyüdükçe (örneğin görüntü işleme projelerindeki büyük boyutlu verilerde) determinant hesaplamak ve polinom kökü bulmak ciddi performans sorunları yaratır. Bu nedenle pratikte, arka planda optimize edilmiş C/Fortran algoritmaları kullanan numpy.linalg.eig fonksiyonu tercih edilmektedir.

Kaynakça:
https://www.geeksforgeeks.org/engineering-mathematics/eigen-values/
https://www.geeksforgeeks.org/machine-learning/ml-linear-algebra-operations/
