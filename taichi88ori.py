
import warp as wp
import warp.render
import numpy as np

wp.init()





gravity = 9.8
bound = 3


# #A simple version of updating
# @wp.kernel
# def update_particles(x: wp.array(dtype=wp.vec3),
#                      v: wp.array(dtype=wp.vec3),
#                      dt: float):
#     tid = wp.tid()
#     p = x[tid]
#     v_curr = v[tid]

#     v_curr.y -= dt * gravity

#     p_new = p + v_curr * dt

#     # bound
#     for i in range(3):
#         if p_new[i] > bound or p_new[i] < -bound:
#             v_curr[i] *= -0.99
#             p_new[i] = min(bound, max(-bound, p_new[i]))

#     x[tid] = p_new
#     v[tid] = v_curr


@wp.func
def outer_product(a: wp.vec3, b: wp.vec3):
    return wp.mat33(
        a.x * b.x, a.x * b.y, a.x * b.z,
        a.y * b.x, a.y * b.y, a.y * b.z,
        a.z * b.x, a.z * b.y, a.z * b.z
    )

@wp.func
def index1Dto3D(idx: int, n: int):
    x = idx % n
    y = (idx // n) % n
    z = idx // (n * n)
    return x, y, z

@wp.kernel
def P2G(x: wp.array(dtype=wp.vec3),
        v: wp.array(dtype=wp.vec3),
        dx:float,
        dt:float,
        grid_v: wp.array(dtype=wp.vec3),
        grid_m: wp.array(dtype=wp.float32),
        n_grid: int,        
        p_vol: float,
        p_mass: float,
        C: wp.array(dtype=wp.mat33),
        J: wp.array(dtype=wp.float32),
        E: float
):  
    ##particles to grid
    p = wp.tid()
    Xp = x[p] / dx
    base = wp.vec3(wp.floor(Xp.x - 0.5), wp.floor(Xp.y - 0.5), wp.floor(Xp.z - 0.5))
    fx = Xp - base
    wx = wp.vec3(0.5 * wp.pow((1.5 - fx.x), 2.0), 0.75 - wp.pow((fx.x - 1.0), 2.0), 0.5 * wp.pow((fx.x - 0.5), 2.0))
    wy = wp.vec3(0.5 * wp.pow((1.5 - fx.y), 2.0), 0.75 - wp.pow((fx.y - 1.0), 2.0), 0.5 * wp.pow((fx.y - 0.5), 2.0))
    wz = wp.vec3(0.5 * wp.pow((1.5 - fx.z), 2.0), 0.75 - wp.pow((fx.z - 1.0), 2.0), 0.5 * wp.pow((fx.z - 0.5), 2.0))
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                offset = wp.vec3(float(i), float(j), float(k))
                dpos = (offset - fx) * dx
                weight = wx[i] * wy[j] * wz[k]
                if 0 <= int(base.x + offset.x) < n_grid and \
                    0 <= int(base.y + offset.y) < n_grid and \
                    0 <= int(base.z + offset.z) < n_grid:
                    idx = int(base.x + offset.x + (base.y + offset.y) * wp.float32(n_grid) + (base.z + offset.z) * wp.float32(n_grid) * wp.float32(n_grid))                        

                    ##PIC
                    grid_v[idx] += weight * v[p]
                    grid_m[idx] += weight

                    ##APIC 
                    # grid_v[idx] += weight * (v[p] + C[p] @ dpos)
                    # grid_m[idx] += weight

                    ##MPM
                    # stress = -dt * 4.0 * E * (J[p] - 1.0) / wp.pow(dx, 2.0)
                    # # stress = -dt * 4.0 * p_vol * E * (J[p] - 1.0) / wp.pow(dx, 2.0)
                    # affine = wp.mat33(
                    #     stress, 0.0, 0.0,
                    #     0.0, stress, 0.0,
                    #     0.0, 0.0, stress) + C[p]
                    # grid_v[idx] += weight * (v[p] + affine @ dpos)
                    # grid_m[idx] += weight 


@wp.kernel
def update_particles(dt:float,
                     grid_v: wp.array(dtype=wp.vec3),
                     grid_m: wp.array(dtype=wp.float32),
                     n_grid: int,
):
    idx = wp.tid()
    i, j, k = index1Dto3D(idx, n_grid)

    if grid_m[idx] > 0:
        inv_m = 1.0 / grid_m[idx]
        grid_v[idx] *= inv_m
    grid_v[idx][1] -= dt * gravity
    if i < bound and grid_v[idx].x <= 0:
        grid_v[idx].x = 0.0
    if i > n_grid - bound and grid_v[idx].x >= 0:
        grid_v[idx].x = 0.0
    if j < bound and grid_v[idx].y <= 0:
        grid_v[idx].y = 0.0
    if j > n_grid - bound and grid_v[idx].y >= 0:
        grid_v[idx].y = 0.0
    if k < bound and grid_v[idx].z <= 0:
        grid_v[idx].z = 0.0
    if k > n_grid - bound and grid_v[idx].z >= 0:
        grid_v[idx].z = 0.0

   
    
@wp.kernel
def G2P(x: wp.array(dtype=wp.vec3),
        v: wp.array(dtype=wp.vec3),
        dx:float,
        dt:float,
        grid_v: wp.array(dtype=wp.vec3),
        grid_m: wp.array(dtype=wp.float32),
        n_grid: int,
        C: wp.array(dtype=wp.mat33),
        J: wp.array(dtype=wp.float32),
):
    p = wp.tid()
    Xp = x[p] / dx
    base = wp.vec3(wp.floor(Xp.x - 0.5), wp.floor(Xp.y - 0.5), wp.floor(Xp.z - 0.5))
    fx = Xp - base
    wx = wp.vec3(0.5 * wp.pow((1.5 - fx.x), 2.0), 0.75 - wp.pow((fx.x - 1.0), 2.0), 0.5 * wp.pow((fx.x - 0.5), 2.0))
    wy = wp.vec3(0.5 * wp.pow((1.5 - fx.y), 2.0), 0.75 - wp.pow((fx.y - 1.0), 2.0), 0.5 * wp.pow((fx.y - 0.5), 2.0))
    wz = wp.vec3(0.5 * wp.pow((1.5 - fx.z), 2.0), 0.75 - wp.pow((fx.z - 1.0), 2.0), 0.5 * wp.pow((fx.z - 0.5), 2.0))
    new_v = wp.vec3(0.0, 0.0, 0.0)
    new_C = wp.mat33(
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                offset = wp.vec3(float(i), float(j), float(k))
                dpos = (offset - fx) * dx
                weight = wx[i] * wy[j] * wz[k]
                if 0 <= int(base.x + offset.x) < n_grid and \
                    0 <= int(base.y + offset.y) < n_grid and \
                    0 <= int(base.z + offset.z) < n_grid:
                    idx = int(base.x + offset.x + (base.y + offset.y) * wp.float32(n_grid) + (base.z + offset.z) * wp.float32(n_grid) * wp.float32(n_grid))
                    g_v = grid_v[idx]
                    new_v += weight * g_v
                    new_C += 4.0 * weight * outer_product(g_v, dpos) / dx
    new_x = x[p] + dt * new_v
    if new_x.x < 0.0 or new_x.x > 1.0:
        new_x.x = min(1.0, max(0.0, new_x.x))
        new_v.x = 0.0
    if new_x.y < 0.0 or new_x.y > 1.0:
        new_x.y = min(1.0, max(0.0, new_x.y))
        new_v.y = 0.0
    if new_x.z < 0.0 or new_x.z > 1.0:
        new_x.z = min(1.0, max(0.0, new_x.z))
        new_v.z = 0.0




    x[p] = new_x
    v[p] = new_v       
    J[p] *= 1.0 + dt * wp.trace(new_C)
    C[p] = new_C










class ParticleSimulation:
    def __init__(self):
        self.frame = 0

        # initialize
        self.n_particles = 8192
        self.n_grid = 128
        self.dx = 1 / self.n_grid
        self.dt = 2e-5

        self.point_radius = 0.01

        self.p_rho = 1
        self.p_vol = 4 * (self.point_radius)**3
        self.p_mass = self.p_vol * self.p_rho


        self.x = wp.array(np.random.uniform(0.1, 0.9, (self.n_particles, 3)), dtype=wp.vec3)
        # self.v = wp.array(np.random.uniform(-0.05, 0.05, (self.n_particles, 3)), dtype=wp.vec3)
        # self.x = wp.array(np.random.uniform(0.2, 0.6, (self.n_particles, 3)), dtype=wp.vec3)
        self.v = wp.array(np.tile([0, -1, 0], (self.n_particles, 1)), dtype=wp.vec3)
        
        self.renderer = wp.render.OpenGLRenderer()

       

        # self.grid_v = wp.HashGrid(self.n_grid, self.n_grid, self.n_grid)
        # self.grid_m = wp.HashGrid(self.n_grid, self.n_grid, self.n_grid)
        self.grid_v = wp.array(np.zeros((self.n_grid * self.n_grid * self.n_grid, 3), dtype=np.float32), dtype=wp.vec3)
        self.grid_m = wp.array(np.zeros((self.n_grid * self.n_grid * self.n_grid), dtype=np.float32), dtype=wp.float32)
        self.grid_cell_size = self.point_radius * 5.0

        self.C = wp.array(np.zeros((self.n_particles, 3, 3), dtype=np.float32), dtype=wp.mat33)  # 3x3矩阵场
        self.J = wp.array(np.ones((self.n_particles,), dtype=np.float32), dtype=wp.float32)  # 标量场
        self.E = 400

    def stimulate(self):
        # self.grid_v = wp.array(np.zeros((self.n_grid * self.n_grid * self.n_grid, 3), dtype=np.float32), dtype=wp.vec3)
        # self.grid_m = wp.array(np.zeros((self.n_grid * self.n_grid * self.n_grid), dtype=np.float32), dtype=wp.float32)

        wp.launch(kernel=P2G,
                    dim=self.n_particles,
                    inputs=[self.x, 
                            self.v, 
                            self.dx, 
                            self.dt, 
                            self.grid_v, 
                            self.grid_m, 
                            self.n_grid,
                            self.p_vol,
                            self.p_mass,
                            self.C,
                            self.J,
                            self.E
                            ])
        
        wp.launch(kernel=update_particles, 
                    dim=self.n_grid, 
                    inputs=[self.dt,
                            self.grid_v,
                            self.grid_m,
                            self.n_grid
                          ])
        
        wp.launch(kernel=G2P,
                    dim=self.n_particles,
                    inputs=[self.x, 
                            self.v, 
                            self.dx, 
                            self.dt, 
                            self.grid_v, 
                            self.grid_m,
                            self.n_grid,
                            self.C,
                            self.J
                            ])
        
    def render(self):

        vertices = self.x.numpy()

        self.renderer.begin_frame(self.frame)
        self.renderer.render_points("ParticleSimulation", vertices, self.point_radius, colors=[(0.2, 0.3, 0.8)] * self.n_particles)
        self.renderer.end_frame()

        self.frame += 1

if __name__ == '__main__':
    sim = ParticleSimulation()
    while True:
        sim.stimulate()
        sim.render()




