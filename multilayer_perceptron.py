

import os
import sys
import dynet as dy
import scipy.sparse as sp
import numpy as np
import pandas as pd
from util import evaluate, load_data
from collections import defaultdict

class MultilayerPerceptronModel():
    
    def __init__(self):
        self.pw = None
        self.pb = None
        self.pv = None
        self.pq = None
        self.pb1 = None
        self.lookup = None
        self.char_dict = None
        self.char_index = None
        self.char_list_len = None
        self.data_len = None
        self.label_list = None
        self.label_len  = None
        self.trainer = None
        self.m = None
        

    def __call__(self,inputs):
        w = dy.parameter(self.pw)
        v = dy.parameter(self.pv)
        b = dy.parameter(self.pb)
        b1 = dy.parameter(self.pb1)
        q = dy.parameter(self.pq)
        lookup = self.lookup
        if len(inputs) == 0:
            inputs = [self.char_list_len]
        lookup_list = [lookup[i] for i in inputs]
        net_input = dy.esum(lookup_list)
        output = dy.softmax(v*dy.tanh(q*dy.dropout(dy.tanh(w*net_input+b),0.3)+b1))
        return output
    
    def gen_loss(self,inputs,expected_answer):
        out = self(inputs)
        loss = -dy.log(dy.pick(out,expected_answer))
        return loss
    
    def gen_best(self,inputs):
        dy.renew_cg()
        output = self(inputs)
        return np.argmax(output.npvalue()) 

    def gen_dict(self,train_data,train_label):
        self.label_list = list(set(train_label))
        self.label_len = len(self.label_list)
        self.data_len = len(train_data)
        char_dict = defaultdict(int)
        char_index = {}
        for i in range(self.data_len):
            for temp_char in train_data[i]:
                char_dict[temp_char] += 1
        self.dict_char = {k:v for k,v in char_dict.items() if v > 1}
        self.char_index = {k:index for index, k in enumerate(self.dict_char.keys())}
        self.char_list_len = len(self.char_index)  

    def phi_x(self,train_data):
        res = [[self.char_index.get(temp_char) for temp_char in temp_data if temp_char in self.char_index]\
               for temp_data in train_data]
        return res
        """for i in range
        mat = sp.dok_matrix((len(train_data),self.char_list_len),dtype=np.int8)
        for i in range(len(train_data)):
            for item in train_data[i]:
                if item in self.char_index:
                    mat[i,self.char_index.get(item)] = 1 
        res = mat.toarray()
        return res"""
    
    def phi_y(self,train_label):
        return [self.label_list.index(train_label[i]) for i in range(len(train_label))]
    
    def train(self, train_data, train_label):
        self.gen_dict(train_data,train_label)
        train_data_fea = self.phi_x(train_data)
        train_label_fea = self.phi_y(train_label)
        self.m = dy.ParameterCollection()
        self.pw = self.m.add_parameters((256,512))
        self.pv = self.m.add_parameters((self.label_len,128))
        self.pb = self.m.add_parameters((256,))
        self.pb1 = self.m.add_parameters((128,))
        self.pq = self.m.add_parameters((128,256))
        self.lookup = self.m.add_lookup_parameters((self.char_list_len+1,512))
        self.trainer = dy.AdamTrainer(self.m)
        count = 0
        accu_loss = []

        epoch = 5
        
        for epoch in range(epoch):
            print("epoch:"+str(epoch))
            for i in range(len(train_data)):
                loss = self.gen_loss(train_data_fea[i],train_label_fea[i])
                accu_loss.append(loss)
                loss_value = loss.value()
                if count%10000 == 0: print(loss_value)
                if count%30 == 0:
                    mean_loss = dy.esum(accu_loss)/float(len(accu_loss))
                    mean_loss.forward()
                    mean_loss.backward()
                    self.trainer.update()
                    dy.renew_cg()
                    accu_loss = []
                count+=1

    def predict(self, model_input):
        input_featurized = self.phi_x(model_input)
        res = [self.gen_best(input_data) for input_data in input_featurized]
        #print(res[:10])
        final_res = [self.label_list[i] for i in res]
        #print(final_res[:10])
        return final_res

if __name__ == "__main__":
    train_data, train_label, dev_data, dev_label, test_data, data_type = load_data(sys.argv)
    
    # Train the model using the training data.
    model = MultilayerPerceptronModel()
    model.train(train_data,train_label)

    train_accuracy = evaluate(model,
                            train_data,train_label,
                            os.path.join("results", "perceptron_" + data_type + "_train_predictions.csv"))
    print(train_accuracy)

    # Predict on the development set. 
    dev_accuracy = evaluate(model,
                            dev_data,dev_label,
                            os.path.join("results", "mlp_" + data_type + "_dev_predictions.csv"))
    print(dev_accuracy)
    seudo_label = np.zeros(len(test_data))
    evaluate(model,
             test_data,seudo_label,
             os.path.join("results", "mlp_" + data_type + "_test_predictions.csv"))
    
