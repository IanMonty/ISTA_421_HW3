import numpy as np
import matplotlib.pyplot as plt

y = 5 ## value for parameter y as defined in assignment
lamb = 7 ## value for parameter lambda as defined in assignment
answer = 0  ## declaring what will become the answer to problem
for x in range(0,y+1):
    e = np.exp(-lamb)
    fact = 1
    for i in range(1,x+1):
        fact = fact * i

##    print(fact)

    num = lamb ** x
    den = fact

    frac = num/den
##    print(frac)
    inst = frac * e

    answer = answer + inst
##    print(y)

print('The probability that y is less than or euqal to ' + str(y) + ' is ' +  str(answer))
print('The probability that y is greater than ' + str(y) + ' is ' + str(1-answer))
