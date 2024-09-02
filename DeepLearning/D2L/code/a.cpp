#include <iostream>
#include <vector>

std::vector<long long> findFactors(long long n) {
    std::vector<long long> factors;
    for (long long i = 1; i <= n; ++i) {
        if (n % i == 0) {
            factors.push_back(i);
        }
    }
    return factors;
}

int main() {
    long long n = 1145141919;
    std::vector<long long> factors = findFactors(n);

    std::cout << "The factors of " << n << " are: ";
    for (long long factor : factors) {
        std::cout << factor << " ";
    }
    std::cout << std::endl;

    return 0;
}
