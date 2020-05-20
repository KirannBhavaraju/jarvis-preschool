import pandas as pd
import numpy as np
import sys

output_location = ''
data = ''
learning_rate = 1
weight_matrix = []
perceptron_output = []
iteration_number = 0
error_calculated = 0
error_matrix = []
x_matrix = []
tsv_data = []
write_list = []

def main():
    global iteration_number,write_list
    format_input()
    gradient_descent()
    write_list = []
    iteration_number = 0
    annealed_gradient_descent()

def perceptron_main(ann=True):
    global data,perceptron_output,error_calculated,weight_matrix,error_matrix,x_matrix,tsv_data,write_list,iteration_number
    tsv_cache_data = pd.read_csv(data, header=None, delimiter='\t')
    tsv_data = tsv_cache_data
    true_classes = (tsv_data[tsv_data.columns[0]]).to_frame()
    x_matrix = tsv_data.dropna(axis=1)
    x_matrix[0] = 1.0
    x_matrix = x_matrix.reindex(sorted(x_matrix.columns), axis = 1)
    if (iteration_number == 0):
        weight_matrix = np.zeros((len(x_matrix.columns[:]), 1))
    sigma_unit = np.matmul(np.matrix(x_matrix), weight_matrix)
    perceptron_output = pd.DataFrame(sigma_unit>0).astype(np.float32)
    error_matrix = np.subtract(pd.DataFrame(true_classes[0] == 'A').astype(dtype=np.float32),perceptron_output)
    error_calculated = np.count_nonzero(pd.DataFrame(true_classes[0] == 'A').astype(np.float32) - perceptron_output)
    write_list.append(error_calculated)
    if(iteration_number==100):
        data_frame = pd.DataFrame(write_list).transpose()
        if ann==True:
            data_frame.to_csv(output_location,sep="\t",index_label=None,index=None,mode='w',header=False)
        if ann==False:
            data_frame.to_csv(output_location,sep="\t",index_label=None,index=None,mode='a',header=False)

def recalculate_weights():
    global weight_matrix,error_matrix,x_matrix,learning_rate
    for index in range(0, len(weight_matrix)):
        weight_matrix[index] = weight_matrix[index] + (learning_rate * np.sum(np.multiply(error_matrix,pd.DataFrame(x_matrix[index]))))

def gradient_descent():
    global iteration_number, learning_rate,write_list
    while(iteration_number<=100):
        perceptron_main()
        recalculate_weights()
        iteration_number+=1

def annealed_gradient_descent():
    global iteration_number, learning_rate,write_list
    while(iteration_number<=100):
        perceptron_main(ann=False)
        iteration_number += 1
        learning_rate = 1.0 / iteration_number
        recalculate_weights()

def format_input():
    global data, output_location
    args = sys.argv[1:]
    for each in range(0, len(args)):
        if args[each] == "--data":
            data = args[each + 1]
        elif args[each] == "--output":
            output_location = args[each+1]
        else: continue

if __name__ == "__main__":
    main()


