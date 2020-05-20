import pandas as pd
import numpy as np
import sys

threshold = ''
data = ''
learning_rate = ''
csv_data_matrix = []
y_true = []
y_predicted = []
weight_matrix = []
output = []
threshold_calculated = 0.0000
iteration_number = 0
prev_SSE = 0.0

def linearReg_main():
    global threshold_calculated , threshold, iteration_number
    while(abs(threshold_calculated) > threshold or iteration_number == 0):
        global prev_SSE
        global csv_data_matrix, weight_matrix , y_predicted , output
        csv_data = pd.read_csv(data, header=None)
        csv_data.insert(0, '0.0', 1)
        csv_data_cache = csv_data  # caching
        y_true = (csv_data[csv_data.columns[-1]]).to_frame()
        x_matrix = csv_data[csv_data.columns[:-1]]
        if(iteration_number == 0):
            weight_matrix = np.zeros((len(x_matrix.columns[:]), 1), dtype=np.float)
        y_predicted = np.matmul(np.matrix(x_matrix),weight_matrix)
        error_matrix = np.subtract(y_true,y_predicted)
        SSE = np.sum(np.square(error_matrix))
        temp_pusher = [iteration_number] + weight_matrix.flatten().tolist() + SSE.tolist()
        output = output + [temp_pusher]
        gradient = []
        for index in range(0, len(x_matrix.columns[:])):
            x_temp = csv_data[csv_data.columns[index]]
            gradient.append(learning_rate * (np.sum(np.matmul(np.matrix(x_temp),np.matrix(error_matrix)))))
        weight_matrix = np.array(np.add(np.matrix(weight_matrix),np.matrix(gradient, dtype=np.float).T), dtype=np.float)
        threshold_calculated = SSE.tolist()[0] - prev_SSE
        prev_SSE = SSE.tolist()[0]
        iteration_number = iteration_number + 1
    write_frame = pd.DataFrame(output)
    write_frame.to_csv("WeightsFinal.csv",index = None , index_label=False, float_format='%.4f',header=False)

if __name__ == '__main__' :
    args = sys.argv[1:]
    for each in range(0, len(args)):
        if args[each] == "--data":
            data = args[each + 1]
        elif args[each] == "--learningRate":
            learning_rate = float(args[each+1])
        elif args[each] == "--threshold":
            threshold = float(args[each+1])
        else: continue
    linearReg_main()