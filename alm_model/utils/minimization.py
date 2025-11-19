import math
from scipy.optimize import root_scalar

def ternary_search(f, left, right):

    while right - left > 2:

        m1 = left + (right - left) // 3
        m2 = right - (right - left) // 3
        if f(m1) < f(m2):
            right = m2
        else:
            left = m1
    
    min_x = left
    min_val = f(left)
    for x in range(left + 1, right + 1):
        if f(x) < min_val:
            min_x = x
            min_val = f(x)
    
    return min_x, min_val

def find_upper_bound(f, x_min, initial_step=1):
    
    x = x_min
    step = initial_step
    current_value = f(x)
    
    while True:
        x_next = x + step
        next_value = f(x_next)
        
        if next_value > current_value:

            return x_next
        
        x = x_next
        current_value = next_value
        step *= 2
        
        # Sécurité: éviter les boucles infinies
        if step > 1e10:
            raise ValueError("The function may not be unimodal.")

def unbounded_minimize_integer(f, x_min, initial_step=1):
    
    x_max = find_upper_bound(f, x_min, initial_step)
    
    return ternary_search(f, x_min, x_max)
