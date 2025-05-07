from scipy import io      # Biblioteca para .mat
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# 1. Carregar o dataset
caminhoArquivos = Path(__file__).resolve().parents[0]
caminhoCompleto   = caminhoArquivos / 'Dataset_Grupo9.mat'
dados             = io.loadmat(str(caminhoCompleto))
valores_strct     = dados['reactionExperiment'][0, 0]

# 2. Extrair tempo (float), entrada e saída
tempo   = valores_strct['sampleTime'].flatten().astype(np.float64)
entrada = valores_strct['dataInput'].flatten()
saida   = valores_strct['dataOutput'].flatten()

# 3. Parâmetros do processo
valor_final = saida[-1]
y1 = 0.283 * valor_final
y2 = 0.632 * valor_final
t1 = tempo[np.where(saida >= y1)[0][0]]
t2 = tempo[np.where(saida >= y2)[0][0]]

tau   = 1.5 * (t2 - t1)         # constante de tempo
theta = t2 - tau               # tempo morto
amplitude_degrau = entrada.mean()
k     = (valor_final - saida[0]) / amplitude_degrau

# 4. Função de transferência nominal (sem atraso)
G_s = ctrl.tf([k], [tau, 1])

# 5. CHR 0% overshoot: cálculo de Kp, Ti, Td
Kp = 0.95 * tau / (k * theta)
Ti = 2.4 * tau
Td = 0.42 * tau

# 6. Monta o controlador PID
PID = ctrl.tf([Kp*Td, Kp, Kp/Ti], [1, 0])

# 7. Aproximação de Pade do atraso e série com o PID em malha
num_pade, den_pade = ctrl.pade(theta, 5)
Pade = ctrl.tf(num_pade, den_pade)
loop = ctrl.series(PID, G_s, Pade)

# 8. Malha fechada e resposta ao degrau
sist_fc = ctrl.feedback(loop, 1)
t_sim, y_sim = ctrl.step_response(sist_fc, T=tempo)

# 9. Plot dos resultados
plt.figure(figsize=(12,6))
plt.plot(tempo, saida,    'k', label='Resposta Real')
plt.plot(t_sim,  y_sim,   'r', label='CHR 0% Overshoot')
plt.title('Controle PID sintonizado pelo CHR (0% Overshoot)')
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura')
plt.legend()
plt.grid()
plt.tight_layout()

# Anotação dos parâmetros
info = ctrl.step_info(sist_fc)
props = dict(boxstyle='round', facecolor='white', alpha=0.6)
txt = (
    f'Kp = {Kp:.3f}\n'
    f'Ti = {Ti:.3f} s\n'
    f'Td = {Td:.3f} s\n'
    f'RiseTime = {info["RiseTime"]:.3f} s\n'
    f'SettlingTime = {info["SettlingTime"]:.3f} s\n'
    f'Overshoot = {info["Overshoot"]:.1f}%'
)
plt.text(tempo[-1]*0.6, max(y_sim)*0.7, txt, bbox=props)
plt.show()

# 10. Resultados no console
print('— PID pelo método CHR (0% Overshoot) —')
print(f'Kp = {Kp:.4f}')
print(f'Ti = {Ti:.4f} s')
print(f'Td = {Td:.4f} s')
print(f"Rise Time   = {info['RiseTime']:.4f} s")
print(f"Settling Time = {info['SettlingTime']:.4f} s")
print(f"Overshoot     = {info['Overshoot']:.2f}%")
