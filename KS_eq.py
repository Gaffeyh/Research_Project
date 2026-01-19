import numpy as np

# -----------------------------
# Spatial grid and initial condition
# -----------------------------
N = 128
x = 32 * np.pi * np.arange(1, N + 1) / N
u = np.cos(x / 16) * (1 + np.sin(x / 16))
v = np.fft.fft(u)

# -----------------------------
# Precompute ETDRK4 quantities
# -----------------------------
h = 1/4                               # time step
k = np.concatenate((np.arange(0, N//2),
                    [0],
                    np.arange(-N//2+1, 0))) / 16
L = k**2 - k**4                       # Fourier multipliers
E = np.exp(h * L)
E2 = np.exp(h * L / 2)

M = 16                                # points for complex means
r = np.exp(1j * np.pi * (np.arange(1, M+1) - 0.5) / M)
LR = h * L[:, None] + r[None, :]

Q = h * np.real(np.mean((np.exp(LR/2) - 1) / LR, axis=1))
f1 = h * np.real(np.mean(
    (-4 - LR + np.exp(LR) * (4 - 3*LR + LR**2)) / LR**3,
    axis=1))
f2 = h * np.real(np.mean(
    (2 + LR + np.exp(LR) * (-2 + LR)) / LR**3,
    axis=1))
f3 = h * np.real(np.mean(
    (-4 - 3*LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3,
    axis=1))

# -----------------------------
# Main time-stepping loop
# -----------------------------
uu = u.copy()
tt = np.array([0.0])

tmax = 150
nmax = round(tmax / h)
nplt = int(np.floor((tmax / 100) / h))

g = -0.5j * k

for n in range(1, nmax + 1):
    t = n * h

    Nv = g * np.fft.fft(np.real(np.fft.ifft(v))**2)
    a = E2 * v + Q * Nv
    Na = g * np.fft.fft(np.real(np.fft.ifft(a))**2)
    b = E2 * v + Q * Na
    Nb = g * np.fft.fft(np.real(np.fft.ifft(b))**2)
    c = E2 * a + Q * (2*Nb - Nv)
    Nc = g * np.fft.fft(np.real(np.fft.ifft(c))**2)

    v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3

    if n % nplt == 0:
        u = np.real(np.fft.ifft(v))
        uu = np.column_stack((uu, u))
        tt = np.append(tt, t)

# uu contains solution snapshots, tt contains corresponding times
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Create meshgrid (MATLAB surf(tt, x, uu))
T, X = np.meshgrid(tt, x)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
surf = ax.plot_surface(
    T, X, uu,
    cmap='autumn',
    linewidth=0,
    antialiased=True
)

# View angle (MATLAB: view([-90 90]))
ax.view_init(elev=90, azim=-90)

# Axis limits (MATLAB: axis tight, set(gca,'zlim',[-5 50]))
ax.set_xlim(tt.min(), tt.max())
ax.set_ylim(x.min(), x.max())
ax.set_zlim(-5, 50)

# Labels (optional but recommended)
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('u')

# Lighting approximation
ax.set_facecolor((0.30, 0.60, 0.60))
fig.colorbar(surf, shrink=0.6, aspect=10)

plt.tight_layout()
plt.show()
