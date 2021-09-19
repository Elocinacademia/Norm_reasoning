n = 3

left = 0
right_down = 2*n-1
up = 0
down = n-1

matrix = [ [0]*n for i in range(n)]
for i in range(1,n**2):
    while i <=n:
        matrix[0][i-1] = i
        i +=1
    while n < i < right_down:
        j = 1
        while j<=n-1:
            matrix[j][n-1] = i
            j+=1
            i+=1
    while right_down =< i < 3*n-2:
        matrix[n-1]
import pdb; pdb.set_trace()




