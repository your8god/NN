from training import trainig, f


data = {
    22: 150,
    23: 155,
    24: 160,
    25: 162,
    26: 171,
    27: 174,
    28: 180,
    29: 183,
    30: 189,
    31: 192,
}

k, c = trainig(data)
print(f'y = {round(k, 2)}x + {round(c, 2)}')

for k, v in data.items():
    print(f'x = {k}, y = {v}, y_r = {f(k)}')
    