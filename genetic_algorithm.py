import numpy as np
import pandas as pd
import math
from extreme_learning_machine import ExtremeLearningMachine

class GeneticAlgorithm:
    def __init__(self, n_feature, gen_size=22, pop_size=70, crossover_rate=0.8, mutation_rate=0.2):
        self.gen = gen_size
        self.pop_size = pop_size
        self.string_size = n_feature
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
    
    def create_population(self): #input: pop_size, string_size
        chromosome = np.random.randint(2, size=(self.pop_size,self.string_size))
        population = pd.DataFrame(chromosome)
        population.columns = pd.MultiIndex.from_product([['chromosome'], population.columns])
        return population
        
    def select_parents(self, population, num_parents): #input: population, num_parents
        idx_parents = np.random.choice(population.shape[0], size=num_parents, replace=False)
        parents = population.iloc[idx_parents]
        return parents
    
    def crossover(self, population): #input: population, crossover_rate, pop_size
        offspring_size = math.ceil(self.crossover_rate * self.pop_size)
        if offspring_size % 2 != 0:
            offspring_size += 1
  
        offspring_crossover = pd.DataFrame()
        for i in range(int(offspring_size/2)):
            parents = self.select_parents(population, 2)
            cut_point = np.random.choice(parents.shape[1])

            offspring = parents.copy()
            offspring['chromosome'].iloc[0][cut_point:] = parents['chromosome'].iloc[1][cut_point:]
            offspring['chromosome'].iloc[1][cut_point:] = parents['chromosome'].iloc[0][cut_point:]
            offspring_crossover = offspring_crossover.append(offspring)

        return offspring_crossover
    
    def mutation(self, population): #input: population, mutation_rate, pop_size
        offspring_size = math.ceil(self.mutation_rate * self.pop_size)

        offspring_mutation = pd.DataFrame()
        for i in range(int(offspring_size)):
            offspring = self.select_parents(population, 1)
            mutation_point = np.random.choice(offspring.shape[1])

            if offspring['chromosome'].iloc[0][mutation_point] == 0:
                offspring['chromosome'].iloc[0][mutation_point] = 1
            else:
                offspring['chromosome'].iloc[0][mutation_point] = 0
            offspring_mutation = offspring_mutation.append(offspring)

        return offspring_mutation
    
    def reproduction(self, population): #input: population
        offspring_crossover = self.crossover(population)
        offspring_mutation = self.mutation(population)
        offspring = pd.concat([offspring_crossover, offspring_mutation], ignore_index=True)
        
        return offspring
    
    def get_data(self, population, data, i): #input: population, data(x_train/x_test), i(index of pop)
        new_data = data.copy()
        for j in range(population.shape[1]):
            if population.iloc[i][j] == 0:
                new_data = new_data.drop(j, axis=1)
        new_data.columns = [i for i in range(0,new_data.shape[1])]
        return new_data
    
    def evaluation(self, population, x_train, x_test, y_train, y_test):
        fitness = []
        predict = []

        for i in range(population.shape[0]):
            x_train_new = self.get_data(population, x_train, i)
            x_test_new = self.get_data(population, x_test, i)
            
            n_newFeature = x_train_new.shape[1]
            elm = ExtremeLearningMachine(n_newFeature) #jumlah fitur baru
            class_predict, accuracy = elm.elm(x_train_new, y_train, x_test_new, y_test)
            
            fitness.append(accuracy)
            predict.append(class_predict)
        return fitness, predict
    
    def selection(self, population, pop_size):
        selected = population.sort_values(by='fitness', ascending=False)
        new_pop = selected.iloc[0:pop_size]

        return new_pop