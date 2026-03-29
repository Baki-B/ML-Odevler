import numpy as np

A = np.array([[4, 1],
              [2, 3]])

print("1. YÖNTEM: Hazır 'eig' Kullanmadan (Matematiksel Yaklaşım)")
iz_A = np.trace(A)
det_A = (A[0,0] * A[1,1]) - (A[0,1] * A[1,0])

katsayilar = [1, -iz_A, det_A]
ozdegerler_manuel = np.roots(katsayilar)
print(f"Hesaplanan Özdeğerler: {ozdegerler_manuel}")

ozvektorler_manuel = []
for l in ozdegerler_manuel:
    M = A - l * np.eye(2)
    v = np.array([-M[0, 1], M[0, 0]])
    
    v_norm = v / np.linalg.norm(v)
    ozvektorler_manuel.append(v_norm)

V_manuel = np.column_stack(ozvektorler_manuel)
print("Hesaplanan Özvektörler (Sütunlar):\n", V_manuel)


print("\n2. YÖNTEM: NumPy 'linalg.eig' Kullanarak")
ozdegerler_numpy, ozvektorler_numpy = np.linalg.eig(A)

print(f"NumPy Özdeğerleri: {ozdegerler_numpy}")
print("NumPy Özvektörleri (Sütunlar):\n", ozvektorler_numpy)