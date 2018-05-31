import numpy as np
import glob

file_list = glob.glob('/Users/jake/Documents/baroclinic_modes/Line_P/canada_DFO/*.txt')

text_file = open(file_list[20], 'r', encoding="ISO-8859-1")

Data = []
for m in range(len(file_list)):
    this_file = file_list[m]
    count0 = 0
    with open(this_file, encoding="ISO-8859-1") as f:
        for _ in range(115):
            next(f)
        for line in f:
            test0 = line.strip().split("\t")
            test1 = test0[0].split()
            count = 0
            data = np.nan * np.zeros((1, len(test1)))
            for i in test1:
                data[0, count] = np.float(i)  # data = one row's worth of data
                count = count + 1

            if count0 < 1:  # deal with first element of storage vs. all others
                data_out = data
            data_out = np.concatenate((data_out, data), axis=0)
            count0 = count0 + 1
    Data.append(data_out)

# todo this works but each file has a different header length


# for l in open(file_list[20], encoding="ISO-8859-1"):
#     test0 = l.strip().split("\t")
#     # tt = text_file.read().strip().split('\t')
#     test1 = test0[0].split()
#     if len(test1) > 6:
#         if count0 < 1:  # deal with first element of storage vs. all others
#             data = np.nan * np.zeros(len(test1))
#         count = 0
#         for i in test1:
#             data[count] = np.float(i)  # data = one row's worth of data
#             count = count + 1
#     count0 = count0 + 1

