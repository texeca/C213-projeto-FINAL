from PyQt5 import QtWidgets  # Base para criar interfaces gráficas (GUI)
from PyQt5.QtWidgets import QFileDialog, QLabel, QGroupBox, QVBoxLayout, QRadioButton, QPushButton,QInputDialog,QMessageBox
from pathlib import Path # Importa a classe Path que serve para manipular caminhos de arquivos/diretórios de forma segura e multiplataforma
import numpy as np # Usada para operações matemáticas e manipulação de arrays/vetores/matrizes
import matplotlib.pyplot as plt # Usado para criar gráficos
import control as ctrl #Usada em engenharia para análise e simulação de sistemas de controle
from scipy import io   #Usado para carregar arquivos .mat do MATLAB

class MethodsTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        # Botão para importar arquivo .mat
        self.btn_import = QPushButton('Importar .mat')
        self.btn_import.clicked.connect(self.import_mat)
        layout.addWidget(self.btn_import)
        self.mat_path = None

        layout.addWidget(QLabel('<h2>Métodos de Sintonia</h2>'))

        # Grupo de rádio
        group = QGroupBox('Escolha um método')
        grp_layout = QVBoxLayout(group)
        self.rb_smith_ol = QRadioButton('Smith - Malha Aberta')
        self.rb_smith_cl = QRadioButton('Smith - Malha Fechada')
        self.rb_sund_ol  = QRadioButton('Sundaresan - Aberto')
        self.rb_sund_cl  = QRadioButton('Sundaresan - Fechado')
        self.rb_comp_smith  = QRadioButton('Comparacao - Smith')
        self.rb_comp_sundaresan  = QRadioButton('Comparacao - Sundaresan')
        
        self.rb_smith_ol.setChecked(True)
        for rb in (self.rb_smith_ol, self.rb_smith_cl, self.rb_sund_ol, self.rb_sund_cl, self.rb_comp_smith, self.rb_comp_sundaresan):
            grp_layout.addWidget(rb)
        layout.addWidget(group)

        # Botões de ação
        self.btn_plot = QPushButton('Plotar Método')
        self.btn_plot.clicked.connect(self.plot_selected_method)
        layout.addWidget(self.btn_plot)
        layout.addStretch()

    def import_mat(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Selecione .mat', filter='MAT files (*.mat)')
        if path:
            self.mat_path = path
            self.btn_import.setText(f'Arquivo: {Path(path).name}')

    def plot_selected_method(self):
        if not self.mat_path:
            QtWidgets.QMessageBox.warning(self, 'Atenção', 'Importe um arquivo .mat primeiro.')
            return

        # Antes de plotar, garantir que figuras antigas estejam fechadas
        plt.close('all')

        if self.rb_smith_ol.isChecked():
            self.plot_smith_aberta()
        elif self.rb_smith_cl.isChecked():
            self.plot_smith_fechada()
        elif self.rb_sund_ol.isChecked():
            self.plot_sund_aberto()
        elif self.rb_sund_cl.isChecked():
            self.plot_sund_fechada()
        elif self.rb_comp_smith.isChecked():
            self.plot_comp_smith()
        else:
            self.plot_comp_sundaresan()

    def plot_sund_aberto(self):
        # Carregar o dataset
        arquivoDados = io.loadmat(self.mat_path)
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
        # Modelo: G(s) = k * exp(-theta*s) / (tau * s + 1)
        def modelo_identificado(k, tau, theta):
            # Função de transferência do sistema de primeira ordem: G(s) = k / (tau * s + 1)
            G_s = ctrl.tf([k], [tau, 1])
            # Aproximação de Pade para o atraso
            num_pade, den_pade = ctrl.pade(theta, 5)  # Aproximação de ordem 5
            Pade_approx = ctrl.tf(num_pade, den_pade)
            # Função de transferência com atraso
            return ctrl.series(G_s, Pade_approx)
        # 6. Calcular a resposta estimada usando o modelo
        resposta_modelo = modelo_identificado(k, tau, theta)
        # 7. Simular a resposta ao degrau do modelo identificado
        t_sim, y_modelo = ctrl.step_response(resposta_modelo*amplitude_degrau, T=tempo)
        # 8. Cálculo do Erro Quadrático Médio (EQM)
        EQM = np.sqrt(np.sum((y_modelo - saida) ** 2) / len(saida))
        # 9. Visualização dos Resultados
        plt.figure(figsize=(12, 6))
        plt.plot(tempo, saida, 'black', label='Resposta Real')
        plt.plot(tempo, entrada, label='Entrada (Degrau)', color='blue')
        plt.plot(t_sim, y_modelo, 'r', label='Modelo Identificado')
        plt.title('Identificação da Planta pelo Método de Sundaresan (Malha Aberta)')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Temperatura ')
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
        #  Posicionar a caixa com os resultados no gráfico
        plt.text(tempo[-1] * 0.77, max(saida) * 0.7, textstr, fontsize=10, bbox=props)
        plt.show() 

    def plot_sund_fechada(self):
        arquivoDados = io.loadmat(self.mat_path)
        valores_strct = arquivoDados['reactionExperiment'][0, 0]
        tempo  = valores_strct['sampleTime'].flatten().astype(float) 
        entrada  = valores_strct['dataInput'].flatten()    
        saida    = valores_strct['dataOutput'].flatten()
        valor_final = saida[-1]
        y1 = 0.353 * valor_final
        y2 = 0.853 * valor_final
        t1 = tempo[np.where(saida >= y1)[0][0]]
        t2 = tempo[np.where(saida >= y2)[0][0]]
        tau = (2/3) * (t2 - t1)
        theta = (1.3*t1) - (0.29*t2)
        amplitude_degrau = entrada.mean()
        k = (valor_final - saida[0]) / amplitude_degrau
        def modelo_identificado(k, tau, theta):
            G_s = ctrl.tf([k], [tau, 1])
            H_s = ctrl.feedback(G_s, 1)
            num_pade, den_pade = ctrl.pade(theta, 5)
            Pade_approx = ctrl.tf(num_pade, den_pade)
            return ctrl.series(G_s, Pade_approx)
        resposta_modelo = modelo_identificado(k, tau, theta)
        t_sim, y_modelo = ctrl.step_response(resposta_modelo*amplitude_degrau, T=tempo)
        EQM = np.sqrt(np.sum((y_modelo - entrada) ** 2) / len(entrada))
        plt.figure(figsize=(12, 6))
        plt.plot(tempo, saida, 'black', label='Resposta Real')
        plt.plot(tempo, entrada, label='Entrada (Degrau)', color='blue')
        plt.plot(t_sim, y_modelo, 'r', label='Modelo Identificado')
        plt.title('Identificação da Planta pelo Método de Sundaresan (Malha Fechada)')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Temperatura')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        props = dict(boxstyle='round', facecolor='white', alpha=0.6)
        textstr = '\n'.join((
            f'Ganho (k): {k:.4f}',
            f'Tempo de Atraso (θ): {theta:.4f} s',
            f'Constante de Tempo (τ): {tau:.4f} s',
            f'(EQM): {EQM:.4f}'))
        plt.text(tempo[-1] * 0.77, max(saida) * 0.7, textstr, fontsize=10, bbox=props)
        plt.show()
#-----------------------------------------------------------------------------------------------------------------------------------     
    def plot_smith_aberta(self):
        arquivoDados = io.loadmat(self.mat_path)
        valores_strct = arquivoDados['reactionExperiment'][0, 0]
        tempo = valores_strct['sampleTime'].flatten().astype(float)
        entrada = valores_strct['dataInput'].flatten()
        saida = valores_strct['dataOutput'].flatten()
        valor_final = saida[-1]
        y1 = 0.283 * valor_final; y2 = 0.632 * valor_final
        t1 = tempo[np.where(saida >= y1)[0][0]]; t2 = tempo[np.where(saida >= y2)[0][0]]
        tau = 1.5 * (t2 - t1); theta = t2 - tau
        amplitude_degrau = entrada.mean()
        k = (valor_final - saida[0]) / amplitude_degrau
        G_s = ctrl.tf([k], [tau, 1])
        num_pade, den_pade = ctrl.pade(theta, 5)
        modelo_estimado = ctrl.series(G_s, ctrl.tf(num_pade, den_pade))
        t_open, y_open = ctrl.step_response(modelo_estimado * amplitude_degrau, T=tempo)
        EQM = np.sqrt(np.mean((y_open - saida) ** 2))
        plt.figure(figsize=(12, 6))
        plt.plot(tempo, saida, 'black', label='Resposta Real do Sistema')
        plt.plot(tempo, entrada, label='Entrada (Degrau)', color='blue')
        plt.plot(t_open, y_open, 'r', label='Modelo Identificado (Smith) Malha Aberta')
        plt.title('Identificação da Planta pelo Método de Smith (Malha Aberta)')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Temperatura')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        props = dict(boxstyle='round', facecolor='white', alpha=0.6)
        textstr = '\n'.join((
            f'Ganho (k): {k:.4f}',
            f'Tempo de Atraso (θ): {theta:.4f} s',
            f'Constante de Tempo (τ): {tau:.4f} s',
            f'(EQM): {EQM:.4f}'))

        plt.text(tempo[-1] * 0.77, max(saida) * 0.7, textstr, fontsize=10, bbox=props)
        plt.show()
#--------------------------------------------------------------------------------------------------------------------
    def plot_smith_fechada(self):
        arquivoDados = io.loadmat(self.mat_path)
        valores_strct = arquivoDados['reactionExperiment'][0, 0]
        tempo = valores_strct['sampleTime'].flatten().astype(float)
        entrada = valores_strct['dataInput'].flatten()
        saida = valores_strct['dataOutput'].flatten()
        valor_final = saida[-1]
        y1 = 0.283 * valor_final; y2 = 0.632 * valor_final
        t1 = tempo[np.where(saida >= y1)[0][0]]; t2 = tempo[np.where(saida >= y2)[0][0]]
        tau = 1.5 * (t2 - t1); theta = t2 - tau
        amplitude_degrau = entrada.mean()
        k = (valor_final - saida[0]) / amplitude_degrau
        G_s = ctrl.tf([k], [tau, 1])
        H_s = ctrl.feedback(G_s, 1)
        resposta_modelo = ctrl.series(H_s, ctrl.tf(*ctrl.pade(theta, 5)))
        t_sim, y_modelo = ctrl.step_response(resposta_modelo * amplitude_degrau, T=tempo)
        EQM = np.sqrt(np.mean((y_modelo - saida) ** 2))
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
        props = dict(boxstyle='round', facecolor='white', alpha=0.6)
        textstr = '\n'.join((
             f'Ganho (k): {k:.4f}',
             f'Tempo de Atraso (θ): {theta:.4f} s',
             f'Constante de Tempo (τ): {tau:.4f} s',
             f'(EQM): {EQM:.4f}'))
        plt.text(tempo[-1] * 0.77, max(saida) * 0.7, textstr, fontsize=10, bbox=props)       
        plt.show()
 #--------------------------------------------------------------------------------------------------------------------  
    def plot_comp_smith(self):
        arquivoDados = io.loadmat(self.mat_path)
        valores_strct = arquivoDados['reactionExperiment'][0, 0]
        tempo  = valores_strct['sampleTime'].flatten().astype(float)
        entrada  = valores_strct['dataInput'].flatten()    
        saida    = valores_strct['dataOutput'].flatten()
        valor_final = saida[-1]
        y1 = 0.283 * valor_final
        y2 = 0.632 * valor_final
        t1 = tempo[np.where(saida >= y1)[0][0]]
        t2 = tempo[np.where(saida >= y2)[0][0]]
        tau = 1.5 * (t2 - t1)
        theta = t2 - tau
        amplitude_degrau = entrada.mean()
        k = (valor_final - saida[0]) / amplitude_degrau
        def modelo_identificado_malha_aberta(k, tau, theta):
            G_s = ctrl.tf([k], [tau, 1])
            num_pade, den_pade = ctrl.pade(theta, 5)
            Pade_approx = ctrl.tf(num_pade, den_pade)
            return ctrl.series(G_s, Pade_approx)
        def modelo_identificado_malha_fechada(k, tau, theta):
            G_s = ctrl.tf([k], [tau, 1])
            H_s = ctrl.feedback(G_s, 1)
            num_pade, den_pade = ctrl.pade(theta, 5)
            Pade_approx = ctrl.tf(num_pade, den_pade)
            return ctrl.series(H_s, Pade_approx)
        resposta_malha_aberta = modelo_identificado_malha_aberta(k, tau, theta)
        resposta_malha_fechada = modelo_identificado_malha_fechada(k, tau, theta)
        t_sim_aberta, y_modelo_aberta = ctrl.step_response(resposta_malha_aberta * amplitude_degrau, T=tempo)
        t_sim_fechada, y_modelo_fechada = ctrl.step_response(resposta_malha_fechada * amplitude_degrau, T=tempo)
        EQM_aberta = np.sqrt(np.sum((y_modelo_aberta - saida) ** 2) / len(saida))
        EQM_fechada = np.sqrt(np.sum((y_modelo_fechada - saida) ** 2) / len(saida))
        info_aberta = ctrl.step_info(resposta_malha_aberta)
        info_fechada = ctrl.step_info(resposta_malha_fechada)
        plt.figure(figsize=(12, 6))
        plt.plot(t_sim_aberta, y_modelo_aberta, 'r', label='Modelo Identificado (Smith) Malha Aberta')
        plt.plot(t_sim_fechada, y_modelo_fechada, 'b', label='Modelo Identificado (Smith) Malha Fechada')
        plt.title('Comparacao entre Malha Aberta e Fechada')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Temperatura')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        props = dict(boxstyle='round', facecolor='white', alpha=0.6)
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
        plt.text(tempo[-1] * 0.68, max(saida) * 0.6, textstr, fontsize=10, bbox=props)
        plt.show()
#--------------------------------------------------------------------------------------------------------------------        
    def plot_comp_sundaresan(self):
        
        arquivoDados = io.loadmat(self.mat_path)
        valores_strct = arquivoDados['reactionExperiment'][0, 0]
        tempo  = valores_strct['sampleTime'].flatten().astype(float) 
        entrada  = valores_strct['dataInput'].flatten()    
        saida    = valores_strct['dataOutput'].flatten()
        valor_final = saida[-1]
        y1 = 0.353 * valor_final
        y2 = 0.853 * valor_final
        t1 = tempo[np.where(saida >= y1)[0][0]]
        t2 = tempo[np.where(saida >= y2)[0][0]]
        tau = (2/3) * (t2 - t1)
        theta = (1.3*t1) - (0.29*t2)
        amplitude_degrau = entrada.mean()  # Amplitude do degrau de entrada
        k = (valor_final - saida[0]) / amplitude_degrau
        def modelo_identificado_malha_aberta(k, tau, theta):
            G_s = ctrl.tf([k], [tau, 1])
            # Aproximação de Pade para o atraso
            num_pade, den_pade = ctrl.pade(theta, 5)  # Aproximação de ordem 5
            Pade_approx = ctrl.tf(num_pade, den_pade)
            # Função de transferência com atraso
            return ctrl.series(G_s, Pade_approx)
        def modelo_identificado_malha_fechada(k, tau, theta):
            G_s = ctrl.tf([k], [tau, 1])
            H_s = ctrl.feedback(G_s, 1)
            num_pade, den_pade = ctrl.pade(theta, 5)  # Aproximação de ordem 5
            Pade_approx = ctrl.tf(num_pade, den_pade)
            return ctrl.series(H_s, Pade_approx)
        resposta_malha_aberta = modelo_identificado_malha_aberta(k, tau, theta)
        resposta_malha_fechada = modelo_identificado_malha_fechada(k, tau, theta)
        t_sim_aberta, y_modelo_aberta = ctrl.step_response(resposta_malha_aberta * amplitude_degrau, T=tempo)
        t_sim_fechada, y_modelo_fechada = ctrl.step_response(resposta_malha_fechada * amplitude_degrau, T=tempo)
        EQM_aberta = np.sqrt(np.sum((y_modelo_aberta - saida) ** 2) / len(saida))
        EQM_fechada = np.sqrt(np.sum((y_modelo_fechada - saida) ** 2) / len(saida))
        info_aberta = ctrl.step_info(resposta_malha_aberta)
        info_fechada = ctrl.step_info(resposta_malha_fechada)
        plt.figure(figsize=(12, 6))
        plt.plot(t_sim_aberta, y_modelo_aberta, 'r', label='Modelo Identificado (Sundaresan) Malha Aberta')
        plt.plot(t_sim_fechada, y_modelo_fechada, 'b', label='Modelo Identificado (Sundaresan) Malha Fechada')
        plt.title('Comparacao entre Malha Aberta e Fechada')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Temperatura')
        plt.legend()
        plt.grid()
        plt.tight_layout()
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
        plt.text(tempo[-1] * 0.68, max(saida) * 0.6, textstr, fontsize=10, bbox=props)
        plt.show()   
 #--------------------------------------------------------------------------------------------------------------------

class PIDTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        # Botão para importar arquivo .mat
        self.btn_import = QPushButton('Importar .mat')
        self.btn_import.clicked.connect(self.import_mat)
        layout.addWidget(self.btn_import)
        self.mat_path = None
        layout.addWidget(QLabel('<h2>Métodos de Sintonia</h2>'))
        # Grupo de rádio
        group = QGroupBox('Escolha um método')
        grp_layout = QVBoxLayout(group)
        self.rb_imc = QRadioButton('IMC')
        self.rb_chr = QRadioButton('CHR - Sem Overshoot')
        
        self.rb_imc.setChecked(True)
        for rb in (self.rb_imc, self.rb_chr):
            grp_layout.addWidget(rb)
        layout.addWidget(group)

        # Botões de ação
        self.btn_plot = QPushButton('Plotar Método')
        self.btn_plot.clicked.connect(self.plot_selected_method)
        layout.addWidget(self.btn_plot)
        layout.addStretch()
        # Antes de plotar, garantir que figuras antigas estejam fechadas
        plt.close('all')
    
    def import_mat(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Selecione .mat', filter='MAT files (*.mat)')
        if path:
            self.mat_path = path
            self.btn_import.setText(f'Arquivo: {Path(path).name}')

    def plot_selected_method(self):
        if not self.mat_path:
            QtWidgets.QMessageBox.warning(self, 'Atenção', 'Importe um arquivo .mat primeiro.')
            return

        # Antes de plotar, garantir que figuras antigas estejam fechadas
        plt.close('all')
        if self.rb_chr.isChecked():
            self.plot_chr()
        elif self.rb_imc.isChecked():
            self.plot_imc()

    def plot_imc(self): 

        arquivoDados = io.loadmat(self.mat_path)
        valores_strct = arquivoDados['reactionExperiment'][0, 0]
        tempo  = valores_strct['sampleTime'].flatten().astype(float)
        entrada  = valores_strct['dataInput'].flatten()    
        saida    = valores_strct['dataOutput'].flatten()
        valor_final = saida[-1]
        y_max = max(saida)
        overshoot = ((y_max - valor_final) / valor_final) * 100
        y1 = 0.353 * valor_final
        y2 = 0.853 * valor_final
        t1 = tempo[np.where(saida >= y1)[0][0]]
        t2 = tempo[np.where(saida >= y2)[0][0]]
        tau = (2/3) * (t2 - t1)
        theta = (1.3*t1) - (0.29*t2)
        amplitude_degrau = entrada.mean()  # Amplitude do degrau de entrada
        k = (valor_final - saida[0]) / amplitude_degrau
        def modelo_identificado(k, tau, theta):
            G_s = ctrl.tf([k], [tau, 1])
            H_s = ctrl.feedback(G_s, 1)
            # Aproximação de Pade para o atraso
            num_pade, den_pade = ctrl.pade(theta, 5)  # Aproximação de ordem 5
            Pade_approx = ctrl.tf(num_pade, den_pade)
            # Função de transferência com atraso
            return ctrl.series(H_s, Pade_approx)
        resposta_modelo = modelo_identificado(k, tau, theta)
        lamb = 100
        kp = ((2*tau)+theta)/(k*((2*lamb)+theta))
        ti = tau+(theta/2)
        td = (tau*theta)/((2*tau)+theta)
        def funcao_PID(kp, ti, td):
            pid = ctrl.tf([kp*td, kp, kp/ti], [1, 0])
            return pid
        PID = funcao_PID(kp, ti, td)
        sistema_em_malha_fechada = ctrl.feedback(ctrl.series(PID, resposta_modelo))
        t_sim, y_modelo = ctrl.step_response(sistema_em_malha_fechada)
        plt.figure(figsize=(12, 6))
        plt.plot(t_sim, y_modelo, 'red', label='PID')
        plt.title('IMC')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Temepratura')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        info = ctrl.step_info(sistema_em_malha_fechada)
        props = dict(boxstyle='round', facecolor='white', alpha=0.6)  # Estilo da caixa
        txt = (
            f"Tempo de subida(tr): {info['RiseTime']:.4f} s\n"
            f"Valor de pico: {info['Peak']:.4f}\n"
            f"Tempo de acomodação(ts): {info['SettlingTime']:.4f} s\n"
            f"Overshoot = {overshoot:.2f}\n"
            f"Kp = {kp:.4f}\n"
            f"Ti = {ti:.4f} s\n"
            f"Td = {td:.4f} s"
            )
# Adicionando os parâmetros identificados no gráfico em uma caixa delimitada
        plt.text(tempo[-1]*0.7, max(y_modelo)*-1.5, txt, bbox=props)
        plt.show()
#---------------------------------------------------------------------------------------------------------
    def plot_chr(self):
        arquivoDados = io.loadmat(self.mat_path)
        valores_strct = arquivoDados['reactionExperiment'][0, 0]
        tempo   = valores_strct['sampleTime'].flatten().astype(np.float64)
        entrada = valores_strct['dataInput'].flatten()
        saida   = valores_strct['dataOutput'].flatten()
        valor_final = saida[-1]
        y1 = 0.353 * valor_final
        y2 = 0.853 * valor_final
        t1 = tempo[np.where(saida >= y1)[0][0]]
        t2 = tempo[np.where(saida >= y2)[0][0]]
        tau = (2/3) * (t2 - t1)
        theta = (1.3*t1) - (0.29*t2)
        amplitude_degrau = entrada.mean()  # Amplitude do degrau de entrada
        k = (valor_final - saida[0]) / amplitude_degrau
        G_s = ctrl.tf([k], [tau, 1])
        Kp = 0.95 * tau / (k * theta)
        Ti = 2.4 * tau
        Td = 0.42 * tau
        PID = ctrl.tf([Kp*Td, Kp, Kp/Ti], [1, 0])
        num_pade, den_pade = ctrl.pade(theta, 5)
        Pade = ctrl.tf(num_pade, den_pade)
        loop = ctrl.series(PID, G_s, Pade)
        sist_fc = ctrl.feedback(loop, 1)
        t_sim, y_sim = ctrl.step_response(sist_fc, T=tempo)
        plt.figure(figsize=(12,6))
        plt.plot(tempo, saida,    'k', label='Resposta Real')
        plt.plot(t_sim,  y_sim,   'r', label='CHR 0% Overshoot')
        plt.title('Controle PID sintonizado pelo CHR (0% Overshoot)')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Temperatura')
        plt.legend()
        plt.grid()
        plt.tight_layout()
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

class ManualTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # 1) Cria o layout e atribui ao widget
        layout = QVBoxLayout()
        self.setLayout(layout)

        # 2) Botão para importar arquivo .mat
        self.btn_import = QPushButton('Importar .mat')
        self.btn_import.clicked.connect(self.import_mat)
        layout.addWidget(self.btn_import)
        self.mat_path = None

        # 3) Título da seção
        layout.addWidget(QLabel('<h2>Manual</h2>'))

        # 4) Botão para plotar método automático
        self.btn_plot_auto = QPushButton('Plotar Automático')
        self.btn_plot_auto.clicked.connect(self.on_plot_auto)
        layout.addWidget(self.btn_plot_auto)

        # 5) Botão para plotar manualmente com valores de Kp, Ti e Td
        self.btn_manual = QPushButton('Manual')
        self.btn_manual.clicked.connect(self.on_manual)
        layout.addWidget(self.btn_manual)

        # 6) Stretch para empurrar os botões para cima
        layout.addStretch()

        # 7) Inicializa variáveis de parâmetros manuais
        self.kp_manual = None
        self.ti_manual = None
        self.td_manual = None
        self.setpoint_manual = None

        # 8) Fecha figuras antigas
        plt.close('all')

    def import_mat(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Selecione .mat', filter='MAT files (*.mat)')
        if path:
            self.mat_path = path
            self.btn_import.setText(f'Arquivo: {Path(path).name}')

    def on_plot_auto(self):
        """Recalcula Kp, Ti, Td pelo método CHR e plota."""
        if not self._check_mat():
            return
        # limpa manual para forçar recálculo
        self.kp_manual = None
        self.setpoint_manual = None
        self.plot_chr()

class ManualTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        # Layout principal
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Botão para importar arquivo .mat
        self.btn_import = QPushButton('Importar .mat')
        self.btn_import.clicked.connect(self.import_mat)
        layout.addWidget(self.btn_import)
        self.mat_path = None

        # Título da seção
        layout.addWidget(QLabel('<h2>Manual</h2>'))

        # Botão para plotar método automático
        self.btn_plot_auto = QPushButton('Plotar Automático')
        self.btn_plot_auto.clicked.connect(self.on_plot_auto)
        layout.addWidget(self.btn_plot_auto)

        # Botão para plotar manualmente com valores de Kp, Ti, Td e Setpoint
        self.btn_manual = QPushButton('Manual')
        self.btn_manual.clicked.connect(self.on_manual)
        layout.addWidget(self.btn_manual)

        layout.addStretch()

        # Inicializa variáveis de parâmetros manuais
        self.kp_manual = None
        self.ti_manual = None
        self.td_manual = None
        self.setpoint_manual = None

        # Fecha figuras antigas
        plt.close('all')

    def import_mat(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Selecione .mat', filter='MAT files (*.mat)')
        if path:
            self.mat_path = path
            self.btn_import.setText(f'Arquivo: {Path(path).name}')

    def on_plot_auto(self):
        """Recalcula Kp, Ti, Td pelo método CHR e plota."""
        if not self._check_mat():
            return
        # limpa valores manuais
        self.kp_manual = None
        self.ti_manual = None
        self.td_manual = None
        self.setpoint_manual = None
        self.plot_chr()

    def on_manual(self):
        """Solicita Kp, Ti, Td e Setpoint manualmente e plota."""
        if not self._check_mat():
            return

        # Valores padrão
        kp_def = self.kp_manual or 0.0
        ti_def = self.ti_manual or 1.0
        td_def = self.td_manual or 0.1
        # Carrega entrada para default setpoint
        arquivo = io.loadmat(self.mat_path)
        vals = arquivo['reactionExperiment'][0, 0]
        entrada = vals['dataInput'].flatten()
        sp_def = self.setpoint_manual or float(entrada.mean())

        # Diálogos para parâmetros
        kp, ok1 = QInputDialog.getDouble(self, 'Kp Manual', 'Digite Kp:', decimals=4, value=kp_def)
        if not ok1: return
        ti, ok2 = QInputDialog.getDouble(self, 'Ti Manual', 'Digite Ti (s):', decimals=4, value=ti_def)
        if not ok2: return
        td, ok3 = QInputDialog.getDouble(self, 'Td Manual', 'Digite Td (s):', decimals=4, value=td_def)
        if not ok3: return
        sp, ok4 = QInputDialog.getDouble(self, 'Setpoint Manual', 'Digite Setpoint:', decimals=4, value=sp_def)
        if not ok4: return

        # Armazena valores
        self.kp_manual = kp
        self.ti_manual = ti
        self.td_manual = td
        self.setpoint_manual = sp
        self.plot_chr()

    def _check_mat(self):
        if not self.mat_path:
            QMessageBox.warning(self, 'Atenção', 'Importe um arquivo .mat primeiro.')
            return False
        return True

    def plot_chr(self):
        # Carrega dados
        arquivoDados = io.loadmat(self.mat_path)
        vals = arquivoDados['reactionExperiment'][0, 0]
        tempo = vals['sampleTime'].flatten().astype(np.float64)
        entrada = vals['dataInput'].flatten()
        saida = vals['dataOutput'].flatten()

        # Calcula curva aberta
        valor_final = saida[-1]
        y1 = 0.353 * valor_final
        y2 = 0.853 * valor_final
        t1 = tempo[np.where(saida >= y1)[0][0]]
        t2 = tempo[np.where(saida >= y2)[0][0]]
        tau = (2/3) * (t2 - t1)
        theta = (1.3 * t1) - (0.29 * t2)

        # Usa o setpoint se definido ou média
        amp = self.setpoint_manual if self.setpoint_manual is not None else entrada.mean()
        k = (valor_final - saida[0]) / amp

        # Define PID
        if self.kp_manual is None:
            Kp = 0.95 * tau / (k * theta)
            Ti = 2.4 * tau
            Td = 0.42 * tau
        else:
            Kp, Ti, Td = self.kp_manual, self.ti_manual, self.td_manual

        # Monta e simula
        G_s = ctrl.tf([k], [tau, 1])
        PID_tf = ctrl.tf([Kp*Td, Kp, Kp/Ti], [1, 0])
        num_pade, den_pade = ctrl.pade(theta, 5)
        Pade = ctrl.tf(num_pade, den_pade)
        loop = ctrl.series(PID_tf, G_s, Pade)
        sist_fc = ctrl.feedback(loop, 1)
        t_sim, y_sim = ctrl.step_response(sist_fc, T=tempo)

        # Plota
        plt.close('all')
        plt.figure(figsize=(12,6))
        plt.plot(tempo, saida, 'k', label='Resposta Real')
        plt.plot(t_sim, y_sim, 'r', label=f'CHR: Kp={Kp:.3f}, Ti={Ti:.3f}, Td={Td:.3f}')
        plt.title('Controle PID Manual')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Temperatura')
        plt.legend()
        plt.grid()
        plt.tight_layout()

        # Caixa de texto
        info = ctrl.step_info(sist_fc)
        props = dict(boxstyle='round', facecolor='white', alpha=0.6)
        txt = (
            f'Kp = {Kp:.3f}\n'
            f'Ti = {Ti:.3f} s\n'
            f'Td = {Td:.3f} s\n'
            f'Setpoint = {self.setpoint_manual if self.setpoint_manual is not None else entrada.mean():.3f}\n'
            f'RiseTime = {info["RiseTime"]:.3f} s\n'
            f'SettlingTime = {info["SettlingTime"]:.3f} s\n'
            f'Overshoot = {info["Overshoot"]:.1f}%'
        )
        plt.text(tempo[-1]*0.6, max(y_sim)*0.2, txt, bbox=props)
        plt.show()
        
    
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Controle PID - Seleção de Métodos')
        self.resize(600, 400)
        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        self.methods_tab = MethodsTab()
        tabs.addTab(self.methods_tab, 'Métodos')

        self.pid_tab = PIDTab()
        tabs.addTab(self.pid_tab, 'PID')

        self.pid_tab = ManualTab()
        tabs.addTab(self.pid_tab, 'Manual')

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
