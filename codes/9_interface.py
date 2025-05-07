import sys
import numpy as np
import scipy.io as io
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QComboBox, QDoubleSpinBox, QCheckBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import control as ctrl

class MethodSelectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Identificação e Controle de Processo")
        self.dataset = None  # t, u, y
        self.k = self.tau = self.theta = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Método selector
        hl = QHBoxLayout()
        hl.addWidget(QLabel("Método:"))
        self.combo = QComboBox()
        self.combo.addItems([
            "Smith Malha Aberta",
            "Smith Malha Fechada",
            "Sundaresan Malha Aberta",
            "Sundaresan Malha Fechada",
            "Comparação Aberta vs Fechada",
            "IMC",
            "CHR Sem Sobrevalor"
        ])
        hl.addWidget(self.combo)
        layout.addLayout(hl)

        # Manual input checkbox
        self.manual_cb = QCheckBox("Inserir manualmente k, τ, θ")
        self.manual_cb.stateChanged.connect(self.toggle_manual)
        layout.addWidget(self.manual_cb)

        # Manual input spinboxes
        manual_layout = QHBoxLayout()
        self.spin_k = QDoubleSpinBox()
        self.spin_k.setPrefix("k: ")
        self.spin_k.setDecimals(4)
        self.spin_tau = QDoubleSpinBox()
        self.spin_tau.setPrefix("τ: ")
        self.spin_tau.setDecimals(4)
        self.spin_theta = QDoubleSpinBox()
        self.spin_theta.setPrefix("θ: ")
        self.spin_theta.setDecimals(4)
        for spin in (self.spin_k, self.spin_tau, self.spin_theta):
            spin.setRange(0, 1e6)
            spin.setEnabled(False)
            manual_layout.addWidget(spin)
        layout.addLayout(manual_layout)

        # Load data button
        btn_load = QPushButton("Carregar .mat")
        btn_load.clicked.connect(self.load_data)
        layout.addWidget(btn_load)

        # Labels for params
        pl = QHBoxLayout()
        self.lbl_k = QLabel("k: N/A")
        self.lbl_tau = QLabel("τ: N/A")
        self.lbl_theta = QLabel("θ: N/A")
        for lbl in (self.lbl_k, self.lbl_tau, self.lbl_theta):
            pl.addWidget(lbl)
        layout.addLayout(pl)

        # Simulate button
        btn_sim = QPushButton("Simular")
        btn_sim.clicked.connect(self.simulate)
        layout.addWidget(btn_sim)

        # Plot area
        self.fig = Figure(figsize=(6,4))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

    def toggle_manual(self):
        manual = self.manual_cb.isChecked()
        for spin in (self.spin_k, self.spin_tau, self.spin_theta):
            spin.setEnabled(manual)

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Selecione o arquivo .mat", "", "MAT files (*.mat)")
        if not path:
            return
        data = io.loadmat(path)
        v = data['reactionExperiment'][0,0]
        t = v['sampleTime'].flatten().astype(np.float64)
        u = v['dataInput'].flatten()
        y = v['dataOutput'].flatten()

        # compute k, tau, theta by Smith for base
        yf = y[-1]
        t1 = t[np.where(y >= 0.283*yf)[0][0]]
        t2 = t[np.where(y >= 0.632*yf)[0][0]]
        tau = 1.5*(t2 - t1)
        theta = t2 - tau
        k = (yf - y[0]) / u.mean()

        self.dataset = (t, u, y)
        self.k, self.tau, self.theta = k, tau, theta
        # update labels and manual spins
        self.lbl_k.setText(f"k: {k:.3f}")
        self.lbl_tau.setText(f"τ: {tau:.3f}")
        self.lbl_theta.setText(f"θ: {theta:.3f}")
        self.spin_k.setValue(k)
        self.spin_tau.setValue(tau)
        self.spin_theta.setValue(theta)

    def simulate(self):
        if self.dataset is None:
            return
        t, u, y = self.dataset
        method = self.combo.currentText()
        # override params if manual, else use stored
        if self.manual_cb.isChecked():
            k = self.spin_k.value()
            tau = self.spin_tau.value()
            theta = self.spin_theta.value()
        else:
            k, tau, theta = self.k, self.tau, self.theta
        # recompute params for methods that change tau/theta
        if method == "Sundaresan Malha Aberta" or method == "Sundaresan Malha Fechada":
            yf = y[-1]
            t1_idx = np.where(y >= 0.353*yf)[0][0]
            t2_idx = np.where(y >= 0.853*yf)[0][0]
            t1 = t[t1_idx]
            t2 = t[t2_idx]
            tau = (2/3)*(t2 - t1)
            theta = 1.3*t1 - 0.29*t2
            k = (yf - y[0]) / u.mean()
        # update labels with current params
        self.lbl_k.setText(f"k: {k:.3f}")
        self.lbl_tau.setText(f"τ: {tau:.3f}")
        self.lbl_theta.setText(f"θ: {theta:.3f}")
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # prepare base models
        G = ctrl.tf([k], [tau, 1])
        num_p, den_p = ctrl.pade(theta, 5)
        P = ctrl.tf(num_p, den_p)

        # implement each method using k, tau, theta
        if method == "Smith Malha Aberta":
            model = ctrl.series(G, P)
            t_sim, y_sim = ctrl.step_response(model * u.mean(), T=t)
            ax.plot(t, y, 'k', label='Real')
            ax.plot(t_sim, y_sim, 'r', label='Smith Aberta')

        elif method == "Smith Malha Fechada":
            H = ctrl.feedback(G, 1)
            model = ctrl.series(H, P)
            t_sim, y_sim = ctrl.step_response(model * u.mean(), T=t)
            ax.plot(t, y, 'k', label='Real')
            ax.plot(t_sim, y_sim, 'b', label='Smith Fechada')

        elif method == "Sundaresan Malha Aberta":
            yf = y[-1]
            t1 = t[np.where(y >= 0.353*yf)[0][0]]
            t2 = t[np.where(y >= 0.853*yf)[0][0]]
            tau_s = 2/3*(t2 - t1)
            theta_s = 1.3*t1 - 0.29*t2
            Gs = ctrl.tf([k], [tau_s,1])
            num_s, den_s = ctrl.pade(theta_s,5)
            Ps = ctrl.tf(num_s, den_s)
            model = ctrl.series(Gs, Ps)
            t_sim, y_sim = ctrl.step_response(model * u.mean(), T=t)
            ax.plot(t, y, 'k', label='Real')
            ax.plot(t_sim, y_sim, 'm', label='Sundaresan Aberta')

        elif method == "Sundaresan Malha Fechada":
            yf = y[-1]
            t1 = t[np.where(y >= 0.353*yf)[0][0]]
            t2 = t[np.where(y >= 0.853*yf)[0][0]]
            tau_s = 2/3*(t2 - t1)
            theta_s = 1.3*t1 - 0.29*t2
            Gs = ctrl.tf([k], [tau_s,1])
            Hs = ctrl.feedback(Gs, 1)
            num_s, den_s = ctrl.pade(theta_s,5)
            Ps = ctrl.tf(num_s, den_s)
            model = ctrl.series(Hs, Ps)
            t_sim, y_sim = ctrl.step_response(model * u.mean(), T=t)
            ax.plot(t, y, 'k', label='Real')
            ax.plot(t_sim, y_sim, 'c', label='Sundaresan Fechada')

        elif method == "Comparação Aberta vs Fechada":
            mo = ctrl.series(G, P)
            to, yo = ctrl.step_response(mo * u.mean(), T=t)
            mc = ctrl.series(ctrl.feedback(G,1), P)
            tc, yc = ctrl.step_response(mc * u.mean(), T=t)
            ax.plot(to, yo, 'r', label='Aberta')
            ax.plot(tc, yc, 'b', label='Fechada')

        elif method == "IMC":
            s = ctrl.tf([1,0],[1])
            C = (tau * s + 1) / (k * tau)
            loop = ctrl.series(C, G, P)
            model = ctrl.feedback(loop, 1)
            t_sim, y_sim = ctrl.step_response(model * u.mean(), T=t)
            ax.plot(t, y, 'k', label='Real')
            ax.plot(t_sim, y_sim, 'g', label='IMC')

        else:  # CHR Sem Sobrevalor
            Kp = 0.95 * tau / (k * theta)
            Ti = 2.4 * tau
            Td = 0.42 * tau
            PID = ctrl.tf([Kp * Td, Kp, Kp / Ti], [1, 0])
            loop = ctrl.series(PID, G, P)
            model = ctrl.feedback(loop, 1)
            t_sim, y_sim = ctrl.step_response(model, T=t)
            ax.plot(t, y, 'k', label='Real')
            ax.plot(t_sim, y_sim, 'r', label='CHR Sem Sobrevalor')

        ax.set_title(method)
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Saída')
        ax.legend()
        ax.grid(True)
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MethodSelectorGUI()
    win.show()
    sys.exit(app.exec_())