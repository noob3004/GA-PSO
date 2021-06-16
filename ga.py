import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import time

DNA_SIZE = 12
POP_SIZE = 100
TIMES = 100
Crossover_probability = 0.4
Mutation_probability = 0.001
NUM_BOUNDER = [0, 15]

# 目的関数
def function(x, y, z):
    return 2*pow(x, 2)-3*pow(y, 2)-4*x+5*y+z

# 遺伝的アルゴリズムのクラス
class GeneticAlgorithm:
    def __init__(self, DNA_SIZE, POP_SIZE, TIMES, Crossover_probability, Mutation_probability,Fitness_method='max') -> None:
        self.dna_size = DNA_SIZE # 遺伝⼦⻑
        self.pop_size = POP_SIZE # 個体数
        self.times = TIMES # 世代数
        self.crossover_prob = Crossover_probability # 交叉確率
        self.mutation_prob = Mutation_probability # 突然変異確率
        self.num_bounder = NUM_BOUNDER # 遺伝子の上限と下限
        self.fitness_method=Fitness_method # 適応性モデル

    # 群の個体の遺伝⼦を初期化
    def init_pop(self):
        pop = np.random.randint(2, size=[self.pop_size, self.dna_size])
        return pop
    # 遺伝⼦の⼆進数から⼗進数へ
    def transform_DNA(self, pop):
        x_pop = pop[:, 0:int(self.dna_size/3)]
        y_pop = pop[:, int(self.dna_size/3):2*int(self.dna_size/3)]
        z_pop = pop[:, 2*int(self.dna_size/3):int(self.dna_size):]
        x = x_pop.dot(2**np.arange(int(self.dna_size/3))[::-1])
        y = y_pop.dot(2**np.arange(int(self.dna_size/3))[::-1])
        z = z_pop.dot(2**np.arange(int(self.dna_size/3))[::-1])
        return x, y, z
    # フィットネス関数
    def get_fitness(self, pop):
        x, y, z = self.transform_DNA(pop)
        fitness_score = function(x, y, z)
        # fitness_modeに依存して、最大値または最小値のフィットネスモードを制御する
        if self.fitness_method == 'max':
            return fitness_score-np.min(fitness_score), fitness_score
        else:
            return -(fitness_score-np.max(fitness_score)),fitness_score
        # 適応度を正の数に変換する

    # 適応度によって⼦個体を選ぶ
    def select(self, pop, fitness_score):
        # 1e-4を加えるのは、適応度が０に⾄ることを防⽌するためです
        fitness_score = fitness_score+1e-4
        idx = np.random.choice(np.arange(
            self.pop_size), size=self.pop_size, replace=True, p=fitness_score**0.5/(fitness_score**0.5).sum())

        # np.random.choiceを使って⼦個体を選ぶ。適応度が⾼ければ⾼いほと確率で選ばれる。    
        return pop[idx]
        

    # 交差関数
    def crossover(self, pop):
        new_pop = []
        for father in pop:
            child = father
            #  ⼀点交差を⾏う
            if np.random.rand() < self.crossover_prob:
                mother = pop[np.random.randint(self.pop_size)]
                cross_point = np.random.randint(DNA_SIZE)
                child[cross_point:] = mother[cross_point:]
            # 突然変異を⾏う
            self.mutation(child)
            new_pop.append(child)
        return new_pop

    # 突然変異関数
    def mutation(self, child):
        if np.random.rand() < self.mutation_prob:
            mutation_point = np.random.randint(
                DNA_SIZE)
            child[mutation_point] = child[mutation_point] ^ 1

    def info(self,fitness):
        if(self.fitness_method=='max'):
            return np.max(fitness)
        else:
            return np.min(fitness)

    # 進化関数
    def evolution(self):
        pop = self.init_pop()
        save=[] # 世代ごとのベストフィット結果の保存
        with trange(self.times) as t:
            for i in t:
                self.crossover(pop) # ⼀点交差と突然変異を⾏う
                fitness, fitness_neo = self.get_fitness(pop)
                t.set_description("enpoch: %i" %i)
                t.set_postfix(result=self.info(fitness_neo))
                save.append(self.info(fitness_neo))
                # time.sleep(0.1)
                pop = self.select(pop, fitness) # 新世代の個体群を構成するための個体群生物の抽出
        return save

    # 結果の可視化関数
    def plot(self,save):
        plt.title("Cross Rate %.3f, Mutation Rate %.3f" % (self.crossover_prob,self.mutation_prob))
        plt.plot(save)
        plt.savefig('./GA_{}'.format(self.fitness_method))
        plt.show()

if __name__ == '__main__':
    # 関数の大きな値を求める
    T = GeneticAlgorithm(DNA_SIZE, POP_SIZE, TIMES,
                         Crossover_probability, Mutation_probability,'max')
    save=T.evolution()
    T.plot(save)
    # 関数の最小値を求める
    T.fitness_method='min'
    save=T.evolution()    
    T.plot(save)

