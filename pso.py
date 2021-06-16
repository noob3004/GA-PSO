import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tgrange
from tqdm.std import trange
import time

pop_size = 100 # 粒子の数
variable_num = 3 # パラメータ数
max_pos = 15 # 最大位置
min_pos = 0 # 最小位置
vel = 0.5 # スピード最大值
max_gen = 100 # 世代数
w = 0.6 # 慣性係数
c1 = 2 # 学習因⼦1
c2 = 2 # 学習因⼦2

# 粒子クラス
class Particle:
    def __init__(self, pop_size, variable_num, max_pos, min_pos, vel) -> None:
        self.pop_x = np.zeros((pop_size, variable_num)) # 粒子の位置
        self.pop_v = np.zeros((pop_size, variable_num)) # 粒子の速度
        self.p_best = np.zeros((pop_size, variable_num)) # 粒子のベスト位置
        self.min_pos = min_pos 
        self.max_pos = max_pos
        self.vel = vel
    
    # 粒子の位置と速度の初期化
    def init_population(self, pop_size, variable_num):
        for i in range(pop_size):
            for j in range(variable_num):
                self.pop_x[i][j] = random.uniform(
                    self.min_pos, self.max_pos)
                self.pop_v[i][j] = random.uniform(-self.vel, self.vel)
            self.p_best[i] = self.pop_x[i]

# 目的関数
def fitness(s,mode='max'):
    x1 = s[0]
    x2 = s[1]
    x3 = s[2]
    y = 2 * x1 ** 2 - 3 * x2 ** 2 - 4 * x1 + 5 * x2 + x3
    if mode=='max':
        return y
    else:
        return -y

# 粒⼦群最適化法クラス
class PSO:
    def __init__(self, max_gen, w, c1, c2,fitness_mode='max') -> None:
        self.max_gen = max_gen
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.g_best = np.zeros(variable_num) # 粒子群のベスト位置
        self.result = [] # 結果の保存
        self.Part = Particle(pop_size, variable_num, max_pos, min_pos, vel) # 初期化
        self.Part.init_population(pop_size, variable_num) 
        # self.get_best()
        self.mode=fitness_mode # 適応値モード
    
    # ベスト位置の初期化
    def get_best(self):
        temp = -1
        for i in range(pop_size):
            fit = fitness(self.Part.p_best[i],self.mode)
            if fit > temp:
                self.g_best = self.Part.p_best[i]
                temp = fit

    # 粒⼦の3次元速度更新
    def update_vel(self, pop_v, pop_x, p_best):
        for i in range(variable_num):
            pop_v[i] = w*pop_v[i]+c1*random.uniform(0, 1)*(
                p_best[i]-pop_x[i])+c2*random.uniform(0, 1)*(self.g_best[i]-pop_x[i])
            if pop_v[i] > self.Part.vel:
                pop_v[i] = self.Part.vel
            elif pop_v[i] < -self.Part.vel:
                pop_v[i] = -self.Part.vel

    # 粒子の3次元位置更新
    def update_pos(self, pop_x, pop_v):
        for i in range(variable_num):
            pop_x[i] = pop_x[i]+pop_v[i]
            if pop_x[i] > self.Part.max_pos:
                pop_x[i] = self.Part.max_pos
            elif pop_x[i] < self.Part.min_pos:
                pop_x[i] = self.Part.min_pos

    # 速度と位置更新する
    def update(self):
        with trange(max_gen) as t:
            for i in t:
                for j in range(pop_size):
                    # 速度更新
                    self.update_vel(
                        self.Part.pop_v[j], self.Part.pop_x[j], self.Part.p_best[j])
                    # 位置更新
                    self.update_pos(self.Part.pop_x[j], self.Part.pop_v[j])

                    # 粒子のベスト位置更新
                    if fitness(self.Part.pop_x[j],self.mode) > fitness(self.Part.p_best[j]):
                        self.Part.p_best[j] = self.Part.pop_x[j]
                    # 粒子群のベスト位置更新
                    if fitness(self.Part.pop_x[j],self.mode) > fitness(self.g_best,self.mode):
                        self.g_best = self.Part.pop_x[j]
                self.result.append(fitness(self.g_best))
                t.set_description("Generation: %i" %i)
                t.set_postfix(fitness=fitness(self.g_best,self.mode))
                # time.sleep(0.1)
            


if __name__ == '__main__':
    T = PSO(max_gen, w, c1, c2,'max')
    T.update()
    #　可視化
    result = T.result
    # T=PSO(max_gen, w, c1, c2,'min')
    # T.update()
    print(T.g_best)
    plt.figure(1)
    plt.title("C1 2.00,C2 2.00,W= 1.00")
    plt.xlabel("epoch", size=14)
    plt.ylabel("velue", size=14)
    plt.plot(result, linewidth=2)
    plt.savefig('./PSO_{}'.format(T.mode))
    plt.show()
