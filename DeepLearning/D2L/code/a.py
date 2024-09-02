import math
def find_factors(n):
    factors = []
    for i in range(1, int(math.sqrt(n + 1)) ):

        if n % i == 0:
            factors.append(i)
            if n//i!=i : factors.append(n//i)
            
            print(i,n//i)
    return factors

# 示例使用
n = 1145141919
factors = find_factors(n)
print(f"数字 {n} 的所有因子是: {factors}")
