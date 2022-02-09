# Import standard library
import numpy as np
import pandas as pd
import math

class ExtremeLearningMachine:

    def __init__(self, input_neuron, hidden_neuron=40):
        
        self.input = input_neuron
        self.hidden = hidden_neuron
    
    #ELM
    #inisialisasi bobot dan bias
    def create_weight_bias(self): #input: hidden neuron, input neuron
        weight = pd.DataFrame(np.random.uniform(-1, 1, (self.hidden, self.input)))
        bias = pd.DataFrame(np.random.uniform(-1, 1, (1, self.hidden)))
        return weight, bias
    
    #transpose matriks
    def transpose_matrix(self, matrix): #input: matriks
        result = pd.DataFrame()
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                result.at[j,i] = matrix.iloc[i][j]
        return result
    
    #perkalian matriks
    def multiplication_matrix(self, matrix1, matrix2): #input: matriks1, matriks2
        result = pd.DataFrame()
        for i in range(matrix1.shape[0]):
            for j in range(matrix2.shape[1]):
                result.at[i,j] = sum(matrix1.iloc[i]*matrix2[j])
        return result
    
    def inverse_matrix(self, matrix): #input: matriks
        inverse = pd.DataFrame()
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i == j:
                    inverse.at[i,j] = 1
                else:
                    inverse.at[i,j] = 0

        for i in range(matrix.shape[0]):
            matrix_copy = matrix.copy()
            for k in range(matrix.shape[0]):
                if i == k:
                    matrix.iloc[k] = matrix.iloc[k] / matrix.iloc[i][i]
                    inverse.iloc[k] = inverse.iloc[k] / matrix_copy.iloc[i][i]
            for k in range(matrix.shape[0]):
                if i != k:
                    matrix.iloc[k] = matrix.iloc[k]-(matrix_copy.iloc[k][i]*matrix.iloc[i])
                    inverse.iloc[k] = inverse.iloc[k]-(matrix_copy.iloc[k][i]*inverse.iloc[i])
        return inverse
    
    #menghitung Hinit
    def count_hinit(self, x_data, weight): #input: x_data, weight
        weight_transpose = self.transpose_matrix(weight)
        hinit = self.multiplication_matrix(x_data, weight_transpose)
        return hinit
    
    #menghitung output hidden layer   
    def count_h(self, hinit, bias): #input: hinit, bias
        h = pd.DataFrame()
        for i in range(hinit.shape[0]):
            for j in range(hinit.shape[1]):
                h.at[i,j] = 1/(1+math.exp(-(hinit.iloc[i][j]+bias.iloc[0][j])))
        return h
    
    def count_moorepenrose(self, h): #input: h
        h_transpose = self.transpose_matrix(h)
        hth = self.multiplication_matrix(h_transpose, h)
        hth_inverse = self.inverse_matrix(hth)
        h_plus = self.multiplication_matrix(hth_inverse, h_transpose)
        return h_plus
    
    #menghitung output weight
    def count_output_weight(self, h_plus, y_train): #input: h_plus, y_train
        output_weight = self.multiplication_matrix(h_plus, y_train)
        return output_weight
    
    #menghitung y prediksi
    def count_y(self, h, output_weight): #input: h, output_weight
        y = self.multiplication_matrix(h, output_weight)
        return y
    
    #menentukan prediksi kelas
    def class_prediction(self, y): #input: y
        y_max = y.max(axis=1)

        class_predict = pd.DataFrame()
        for i in range(y.shape[0]):
            temp = 0
            for j in range(y.shape[1]):
                if y.iloc[i][j] == y_max.iloc[i]:
                    temp = j+1
            class_predict.at[i,0] = temp
        return class_predict 
    
    def training(self, x_train, y_train, weight, bias): #x_train, weight, bias, y_train
        hinit = self.count_hinit(x_train, weight)
        h = self.count_h(hinit, bias)
        h_plus = self.count_moorepenrose(h)
        output_weight = self.count_output_weight(h_plus, y_train)
        return output_weight
    
    def testing(self, x_test, output_weight, weight, bias): #x_test, weight, bias, output_weight
        hinit = self.count_hinit(x_test, weight)
        h = self.count_h(hinit, bias)
        y = self.count_y(h, output_weight)
        return y
    
    def accuracy(self, class_predict, y_test): #class_prediksi, y_test
        true = 0
        for i in range(class_predict.shape[0]):
            if class_predict.iloc[i][0] == y_test.iloc[i]['Label']:
                true +=1
        accuracy = true/y_test.shape[0]*100
        return accuracy
    
    def elm(self, x_train, y_train, x_test, y_test): #x_train, y_train, x_test, y_test
        weight, bias = self.create_weight_bias()
        output_weight = self.training(x_train, y_train, weight, bias)
        y = self.testing(x_test, output_weight, weight, bias)
        class_predict = self.class_prediction(y)
        accuracy = self.accuracy(class_predict, y_test)
        return class_predict, accuracy