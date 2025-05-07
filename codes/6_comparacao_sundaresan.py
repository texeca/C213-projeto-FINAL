from scipy import io      # Biblioteca para manipulação dos arquivos.mat.
from pathlib import Path  # Biblioteca para manipulação dos diretórios.
import pandas as pd
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# Carregar o dataset
caminhoArquivos = Path(__file__).resolve().parents[0]
caminhoCompleto = str(Path(caminhoArquivos/'Dataset_Grupo9'))
arquivoDados = io.loadmat(caminhoCompleto)

valores_strct = arquivoDados['reactionExperiment'][0, 0]

# Extraindo entrada, saída e tempo
tempo  = valores_strct['sampleTime'].flatten().astype(float) 
entrada  = valores_strct['dataInput'].flatten()    
saida    = valores_strct['dataOutput'].flatten()
# 1. Determinar o valor final da saída
valor_final = saida[-1]

# 2. Encontrar os tempos correspondentes a 35,3% e 85,3% do valor final
y1 = 0.353 * valor_final
y2 = 0.853 * valor_final

# Encontrar t1 e t2 nos dados
t1 = tempo[np.where(saida >= y1)[0][0]]
t2 = tempo[np.where(saida >= y2)[0][0]]

# 3. Calcular τ e θ usando o Método de Sundaresan
tau = (2/3) * (t2 - t1)
theta = (1.3*t1) - (0.29*t2)

# 4. Calcular o ganho k
amplitude_degrau = entrada.mean()  # Amplitude do degrau de entrada
k = (valor_final - saida[0]) / amplitude_degrau

# 5. Modelo Identificado usando a Função de Transferência
def modelo_identificado_malha_aberta(k, tau, theta):
    G_s = ctrl.tf([k], [tau, 1])
    # Aproximação de Pade para o atraso
    num_pade, den_pade = ctrl.pade(theta, 5)  # Aproximação de ordem 5
    Pade_approx = ctrl.tf(num_pade, den_pade)
    # Função de transferência com atraso
    return ctrl.series(G_s, Pade_approx)

# 6. Modelo Identificado usando a Função de Transferência
def modelo_identificado_malha_fechada(k, tau, theta):
    G_s = ctrl.tf([k], [tau, 1])
    H_s = ctrl.feedback(G_s, 1)
    # Aproximação de Pade para o atraso
    num_pade, den_pade = ctrl.pade(theta, 5)  # Aproximação de ordem 5
    Pade_approx = ctrl.tf(num_pade, den_pade)
    # Função de transferência com atraso
    return ctrl.series(H_s, Pade_approx)

# 7. Calcular a resposta estimada usando o modelo
resposta_malha_aberta = modelo_identificado_malha_aberta(k, tau, theta)
resposta_malha_fechada = modelo_identificado_malha_fechada(k, tau, theta)

# 8. Simular a resposta ao degrau dos modelos
t_sim_aberta, y_modelo_aberta = ctrl.step_response(resposta_malha_aberta * amplitude_degrau, T=tempo)
t_sim_fechada, y_modelo_fechada = ctrl.step_response(resposta_malha_fechada * amplitude_degrau, T=tempo)

# 9. Calcular o Erro Quadrático Médio (EQM) para ambos os modelos
EQM_aberta = np.sqrt(np.sum((y_modelo_aberta - saida) ** 2) / len(saida))
EQM_fechada = np.sqrt(np.sum((y_modelo_fechada - saida) ** 2) / len(saida))

info_aberta = ctrl.step_info(resposta_malha_aberta)
info_fechada = ctrl.step_info(resposta_malha_fechada)

# 10. Visualização dos Resultados malha aberta
plt.figure(figsize=(12, 6))
plt.plot(t_sim_aberta, y_modelo_aberta, 'r', label='Modelo Identificado (Sundaresan) Malha Aberta')
plt.plot(t_sim_fechada, y_modelo_fechada, 'b', label='Modelo Identificado (Sundaresan) Malha Fechada')
plt.title('Comparacao entre Malha Aberta e Fechada')
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura')
plt.legend()
plt.grid()
plt.tight_layout()

# Adicionando os parâmetros identificados no gráfico em uma caixa delimitada
props = dict(boxstyle='round', facecolor='white', alpha=0.6)  # Estilo da caixa

textstr = '\n'.join([
    f"Tempo de subida (Malha Aberta): {info_aberta['RiseTime']:.4f} s",
    f"Tempo de acomodação (Malha Aberta): {info_aberta['SettlingTime']:.4f} s",
    f"Valor final (Malha Aberta): {info_aberta['Peak']:.4f}",
    f"Erro Quadrático Médio (Aberta): {EQM_aberta:.4f}",
    f"\nTempo de subida (Malha Fechada): {info_fechada['RiseTime']:.4f} s",
    f"Tempo de acomodação (Malha Fechada): {info_fechada['SettlingTime']:.4f} s",
    f"Valor final (Malha Fechada): {info_fechada['Peak']:.4f}",
    f"Erro Quadrático Médio (Fechada): {EQM_fechada:.4f}"
])


# Posicionar a caixa com os resultados no gráfico
plt.text(tempo[-1] * 0.68, max(saida) * 0.6, textstr, fontsize=10, bbox=props)

plt.show()