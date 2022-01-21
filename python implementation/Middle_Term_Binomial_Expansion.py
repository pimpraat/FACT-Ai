# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import math
 

# factorial of a number
def factorial(n) :
     
    fact = 1
    for i in range(1, n+1) :
        fact = fact * i
 
    return fact;
 
# Function to find middle term in
# binomial expansion series.
def findMiddleTerm(A, X, n) :
 
    if (n % 2 == 0) :
         
        # If n is even
         
        # calculating the middle term
        i = int(n / 2)
 
        # calculating the value of A to
        # the power k and X to the power k
        aPow = int(math.pow(A, n - i))
        xPow = int(math.pow(X, i))
 
        middleTerm1 = ((math.factorial(n) /
                       (math.factorial(n - i)
                       * math.factorial(i)))
                       * aPow * xPow)
                                 
        print ("MiddleTerm = {}" .
                     format(middleTerm1))
 
    else :
 
        # If n is odd
         
        # calculating the middle term
        i = int((n - 1) / 2)
        j = int((n + 1) / 2)
 
        # calculating the value of A to the
        # power k and X to the power k
        aPow = int(math.pow(A, n - i))
        xPow = int(math.pow(X, i))
 
        middleTerm1 = ((math.factorial(n)
                    / (math.factorial(n - i)
                    * math.factorial(i)))
                        * aPow * xPow)
 
        # calculating the value of A to the
        # power k and X to the power k
        aPow = int(math.pow(A, n - j))
        xPow = int(math.pow(X, j))
 
        middleTerm2 = ((math.factorial(n)
                   / (math.factorial(n - j)
                   * math.factorial(j)))
                      * aPow * xPow)
 
        print ("MiddleTerm1 = {}" .
               format(int(middleTerm1)))
                         
        print ("MiddleTerm2 = {}" .
               format(int(middleTerm2)))

n = 100
A = 200
X = 300
 
# function call
findMiddleTerm(A, X, n)

# %%
