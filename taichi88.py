
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

@wp.kernel
def update_particles(x: wp.array(dtype=wp.vec3),
                     v: wp.array(dtype=wp.vec3),
                     dx:float,
                     dt:float,
                     grid_v: wp.array(dtype=wp.vec3),
                     grid_m: wp.array(dtype=wp.float32),
                     n_particles: int,
                     n_grid: int,
                     p_vol: float,
                     p_mass: float,
                     C: wp.array(dtype=wp.mat33),
                     J: wp.array(dtype=wp.float32),
                     E: float
):  
    ##P2G
    for p in range(n_particles):
        Xp = x[p] / dx
        base = wp.vec3(wp.floor(Xp.x - 0.5), wp.floor(Xp.y - 0.5), wp.floor(Xp.z - 0.5))
        fx = Xp - base
        wx = wp.vec3(0.5 * wp.pow((1.5 - fx.x), 2.0), 0.75 - wp.pow((fx.x - 1.0), 2.0), 0.5 * wp.pow((fx.x - 0.5), 2.0))
        wy = wp.vec3(0.5 * wp.pow((1.5 - fx.y), 2.0), 0.75 - wp.pow((fx.y - 1.0), 2.0), 0.5 * wp.pow((fx.y - 0.5), 2.0))
        wz = wp.vec3(0.5 * wp.pow((1.5 - fx.z), 2.0), 0.75 - wp.pow((fx.z - 1.0), 2.0), 0.5 * wp.pow((fx.z - 0.5), 2.0))
        # stress = -dt * 4.0 * E * p_vol * (J[p] - 1.0) / wp.pow(dx, 2.0)
        # affine = wp.mat33(
        #     stress, 0.0, 0.0,
        #     0.0, stress, 0.0,
        #     0.0, 0.0, stress) + p_mass * C[p]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    offset = wp.vec3(float(i), float(j), float(k))
                    # dpos = (offset - fx) * dx
                    weight = wx[i] * wy[j] * wz[k]
                    idx = int(base.x + offset.x + (base.y + offset.y) * wp.float32(n_grid) + (base.z + offset.z) * wp.float32(n_grid) * wp.float32(n_grid))
                    # grid_v[idx] += weight * (p_mass * v[p] + affine @ dpos)
                    # grid_v[idx] += weight * (p_mass * v[p])
                    grid_v[idx] += weight * v[p]
                    # grid_m[idx] += weight * p_mass
                    grid_m[idx] += weight

    for i in range(n_grid):
        for j in range(n_grid):
            for k in range(n_grid):
                idx = i + j * n_grid + k * n_grid * n_grid

                if grid_m[idx] > 0:
                    inv_m = 1.0 / grid_m[idx]
                    grid_v[idx] *= inv_m
                grid_v[idx][1] -= dt * gravity
                if i < bound and grid_v[idx].x < 0:
                    grid_v[idx].x = 0.0
                if i > n_grid - bound and grid_v[idx].x > 0:
                    grid_v[idx].x = 0.0
                if j < bound and grid_v[idx].y < 0:
                    grid_v[idx].y = 0.0
                if j > n_grid - bound and grid_v[idx].y > 0:
                    grid_v[idx].y = 0.0
                if k < bound and grid_v[idx].z < 0:
                    grid_v[idx].z = 0.0
                if k > n_grid - bound and grid_v[idx].z > 0:
                    grid_v[idx].z = 0.0
                
    
    ##G2P
    for p in range(n_particles):
        Xp = x[p] / dx
        base = wp.vec3(wp.floor(Xp.x - 0.5), wp.floor(Xp.y - 0.5), wp.floor(Xp.z - 0.5))
        fx = Xp - base
        wx = wp.vec3(0.5 * wp.pow((1.5 - fx.x), 2.0), 0.75 - wp.pow((fx.x - 1.0), 2.0), 0.5 * wp.pow((fx.x - 0.5), 2.0))
        wy = wp.vec3(0.5 * wp.pow((1.5 - fx.y), 2.0), 0.75 - wp.pow((fx.y - 1.0), 2.0), 0.5 * wp.pow((fx.y - 0.5), 2.0))
        wz = wp.vec3(0.5 * wp.pow((1.5 - fx.z), 2.0), 0.75 - wp.pow((fx.z - 1.0), 2.0), 0.5 * wp.pow((fx.z - 0.5), 2.0))
        new_v = wp.vec3(0.0, 0.0, 0.0)
        # new_C = wp.mat33(
        #     0.0, 0.0, 0.0,
        #     0.0, 0.0, 0.0,
        #     0.0, 0.0, 0.0)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    offset = wp.vec3(float(i), float(j), float(k))
                    # dpos = (offset - fx) * dx
                    weight = wx[i] * wy[j] * wz[k]
                    idx = int(base.x + offset.x + (base.y + offset.y) * wp.float32(n_grid) + (base.z + offset.z) * wp.float32(n_grid) * wp.float32(n_grid))
                    new_v += weight * grid_v[idx]
                    # new_C += 4.0 * weight * outer_product(g_v, dpos) / wp.pow(dx, 2.0)
        new_x = x[p] + dt * v[p]
        if new_x.x < 0.0 or new_x.x > 1.0:
            new_x.x = min(1.0, max(0.0, new_x.x))
            new_v.x *= -0.99
        if new_x.y < 0.0 or new_x.y > 1.0:
            new_x.y = min(1.0, max(0.0, new_x.y))
            new_v.y *= -0.99
        if new_x.z < 0.0 or new_x.z > 1.0:
            new_x.z = min(1.0, max(0.0, new_x.z))
            new_v.z *= -0.99




        x[p] = new_x
        v[p] = new_v       
        # J[p] *= 1.0 + dt * wp.trace(new_C)
        # C[p] = new_C








class ParticleSimulation:
    def __init__(self):
        self.frame = 0

        # initialize
        self.n_particles = 8192
        self.n_grid = 128
        self.dx = 1 / self.n_grid
        self.dt = 2e-5

        self.p_rho = 1
        self.p_vol = (self.dx * 0.5)**2
        self.p_mass = self.p_vol * self.p_rho

        self.x = wp.array(np.random.uniform(0.1, 0.9, (self.n_particles, 3)), dtype=wp.vec3)
        self.v = wp.array(np.random.uniform(-0.05, 0.05, (self.n_particles, 3)), dtype=wp.vec3)
        self.renderer = wp.render.OpenGLRenderer()

        self.point_radius = 0.01

        # self.grid_v = wp.HashGrid(self.n_grid, self.n_grid, self.n_grid)
        # self.grid_m = wp.HashGrid(self.n_grid, self.n_grid, self.n_grid)
        self.grid_v = wp.array(np.zeros((self.n_grid * self.n_grid * self.n_grid, 3), dtype=np.float32), dtype=wp.vec3)
        self.grid_m = wp.array(np.zeros((self.n_grid * self.n_grid * self.n_grid), dtype=np.float32), dtype=wp.float32)
        self.grid_cell_size = self.point_radius * 5.0

        self.C = wp.array(np.zeros((self.n_particles, 3, 3), dtype=np.float32), dtype=wp.mat33)  # 3x3矩阵场
        self.J = wp.array(np.ones((self.n_particles,), dtype=np.float32), dtype=wp.float32)  # 标量场
        self.E = 400

    def render(self):

        # wp.launch(kernel=update_particles, dim=self.n_particles, inputs=[self.x, self.v, self.dt])
        wp.launch(kernel=update_particles, 
                  dim=self.n_grid, 
                  inputs=[self.x, 
                          self.v,
                          self.dx, 
                          self.dt, 
                          self.grid_v, 
                          self.grid_m, 
                          self.n_particles, 
                          self.n_grid,
                          self.p_vol,
                          self.p_mass,
                          self.C,
                          self.J,
                          self.E
                          ])

        vertices = self.x.numpy()

        self.renderer.begin_frame(self.frame)
        self.renderer.render_points("ParticleSimulation", vertices, self.point_radius, colors=[(1.0, 1.0, 1.0)] * self.n_particles)
        self.renderer.end_frame()

        self.frame += 1


if __name__ == '__main__':
    sim = ParticleSimulation()
    while True:
        sim.render()























# import warp as wp
# import warp.render
# import numpy as np

# wp.init()

# n_particles = 8192
# n_grid = 128
# dx = 1 / n_grid
# dt = 2e-4

# p_rho = 1
# p_vol = (dx * 0.5)**2
# p_mass = p_vol * p_rho
# gravity = 9.8
# bound = 3
# E = 400




# @wp.kernel
# def update_particles(x: wp.array(dtype=wp.vec3),
#                     v: wp.array(dtype=wp.vec3),
#                     dt: float):
#         tid = wp.tid()
#         p = x[tid]
#         v[tid].y -= dt * gravity  # 应用重力
#         p_new = p + v[tid] * dt  # 更新粒子位置

#         for i in range(3):
#             if p_new[i] > 1.0 or p_new[i] < -1.0:
#                 v[i] *= -0.99
#                 p_new[i] = min(1.0, max(-1.0, p_new[i]))

#         x[tid] = p_new

#         # # 边界处理
#         # for i in range(2):
#         #     if p_new[i] < 0:
#         #        v[tid][i] = -v[tid][i]
#         #        p_new[i] = 0
#         #     if p_new[i] > 1:
#         #         v[tid][i] = -v[tid][i]
#         #         p_new[i] = 1
#         #         x[tid] = p_new  # 更新粒子位置

# class taichi88:
#     def __init__(self):
#         self.frame = 0

#         self.x = wp.array(np.random.uniform(-1, 1, (n_particles, 3)), dtype=wp.vec3)
#         self.v = wp.array(np.random.uniform(-3, 3, (n_particles, 3)), dtype=wp.vec3)

#         self.C = wp.array(np.zeros((n_particles, 3, 3), dtype=np.float32), dtype=wp.mat33)
#         self.J = wp.array(np.ones((n_particles,), dtype=np.float32), dtype=wp.float32)

#         self.grid_v = wp.array(np.zeros((n_grid, n_grid, 3), dtype=np.float32), dtype=wp.vec3)
#         self.grid_m = wp.array(np.zeros((n_grid, n_grid), dtype=np.float32), dtype=wp.float32)


#         self.renderer = wp.render.OpenGLRenderer()
#         self.vertices = self.x.numpy()
#         self.indices = np.arange(len(self.vertices), dtype = np.int32)
#         self.radius: float = 0.02

#     def render(self):

#     # 调用核函数更新粒子
#         wp.launch(kernel=update_particles, dim=n_particles, inputs=[self.x, self.v, dt])

#         self.renderer.begin_frame(self.frame)

#     # 渲染粒子
#         self.renderer.render_points("MPM88", self.vertices, self.radius, colors=[(1.0, 1.0, 1.0)] * n_particles)

#         self.renderer.end_frame()

#     # 更新帧数
#         self.frame += 1

# if __name__ == '__main__':
#     Example = taichi88()
#     while True:
#         Example.render()













