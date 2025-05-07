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

# 2. Encontrar os tempos correspondentes a 28,3% e 63,2% do valor final
y1 = 0.283 * valor_final
y2 = 0.632 * valor_final

# Encontrar t1 e t2 nos dados
t1 = tempo[np.where(saida >= y1)[0][0]]
t2 = tempo[np.where(saida >= y2)[0][0]]

# 3. Calcular τ e θ usando o Método de Smith
tau = 1.5 * (t2 - t1)
theta = t2 - tau

# 4. Calcular o ganho k
amplitude_degrau = entrada.mean()  # Amplitude do degrau de entrada
k = (valor_final - saida[0]) / amplitude_degrau

# 5. Modelo Identificado usando a Função de Transferência
# Modelo: G(s) = k * exp(-theta*s) / (tau * s + 1)

# Função de transferência do sistema de primeira ordem: G(s) = k / (tau * s + 1)
G_s = ctrl.tf([k], [tau, 1])
H_s = ctrl.feedback(G_s, 1)
# Aproximação de Pade para o atraso
num_pade, den_pade = ctrl.pade(theta, 5)  # Aproximação de ordem 5
Pade_approx = ctrl.tf(num_pade, den_pade)

# 6. Calcular a resposta estimada usando o modelo
resposta_modelo = ctrl.series(H_s, Pade_approx)

# 7. Simular a resposta ao degrau do modelo identificado
t_sim, y_modelo = ctrl.step_response(resposta_modelo*amplitude_degrau, T=tempo)

# 8. Cálculo do Erro Quadrático Médio (EQM)
EQM = np.sqrt(np.sum((y_modelo - entrada) ** 2) / len(entrada))

# 9. Visualização dos Resultados
plt.figure(figsize=(12, 6))
plt.plot(tempo, saida, 'black', label='Resposta Real')
plt.plot(tempo, entrada, label='Entrada (Degrau)', color='blue')
plt.plot(t_sim, y_modelo, 'r', label='SmModelo Identificado')
plt.title('Identificação da Planta pelo Método de Smith (Malha Fechada)')
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura')
plt.legend()
plt.grid()
plt.tight_layout()

# Adicionando os parâmetros identificados no gráfico em uma caixa delimitada
props = dict(boxstyle='round', facecolor='white', alpha=0.6)  # Estilo da caixa

textstr = '\n'.join((
     f'Ganho (k): {k:.4f}',
     f'Tempo de Atraso (θ): {theta:.4f} s',
     f'Constante de Tempo (τ): {tau:.4f} s',
     f'(EQM): {EQM:.4f}'))

# # Posicionar a caixa com os resultados no gráfico
plt.text(tempo[-1] * 0.77, max(saida) * 0.7, textstr, fontsize=10, bbox=props)

plt.show()

# Exibir os resultados
print(f'Método de Identificação: Smith (Malha Fechada)')
print(f'Parâmetros Identificados:')
print(f'Ganho (k): {k:.4f}')
print(f'Tempo de Atraso (θ): {theta:.4f} s')
print(f'Constante de Tempo (τ): {tau:.4f} s')
print(f'Erro Quadrático Médio (EQM): {EQM}')

info = ctrl.step_info(resposta_modelo)
# Exibir o tempo de subida e o tempo de acomodação
print(f"Tempo de subida(tr): {info['RiseTime']:.4f} s")
print(f"Tempo de acomodação(ts): {info['SettlingTime']:.4f} s")
print(f"valor de pico: {info['Peak']:.4f}")
