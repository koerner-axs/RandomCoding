#include <complex>
#include <vector>
#include <iostream>
#include <numbers>


using complex = std::complex<double>;
using namespace std::complex_literals;

#define PI std::numbers::pi


// n² runtime polynomial multiplication algorithm for testing
std::vector<double> test_multiply(const std::vector<double>& p1, const std::vector<double>& p2) {
	std::vector<double> result((int)(p1.size() + p2.size()) - 1, 0.0);

	for (int i1 = 0; i1 < (int)p1.size(); i1++) {
		for (int i2 = 0; i2 < (int)p2.size(); i2++) {
			result[i1 + i2] += p1[i1] * p2[i2];
		}
	}

	return result;
}


// FFT-based n * log(n) fourier transform implementation.
// This function computes the (inverse) discrete fourier transform of the given signal
// (or frequency spectrum).
// 
// For the forward case, this function computes:
// X_k = \sum_{n=0}^{N-1} x_n \cdot \exp(-2\pi i \cdot \frac{nk}{N}).
// For the inverse case, it computes:
// X_k = \frac{1}{N} \sum_{n=0}^{N-1} x_n \cdot \exp(2\pi i \cdot \frac{nk}{N}).
std::vector<complex> _fft_impl(const std::vector<complex>& signal, bool inverse,
		int points, int start_index, int stride) {

	if (points == 2) {
		// Forward:
		// A_even(X²) + X * A_odd(X²) with X = e^(-i*pi*2) = 1
		// A_even(X²) + X * A_odd(X²) with X = e^(-i*pi) = -1
		// Inverse:
		// A_even(X²) + X * A_odd(X²) with X = e^(i*pi*2) = 1
		// A_even(X²) + X * A_odd(X²) with X = e^(i*pi) = -1
		// Therefore no distinction is necessary.
		return { signal[start_index] + signal[start_index + stride],
				 signal[start_index] - signal[start_index + stride] };
	}

	// Recursion
	std::vector<complex> even_parts = _fft_impl(signal, inverse, points / 2, start_index, stride * 2);
	std::vector<complex> odd_parts = _fft_impl(signal, inverse, points / 2, start_index + stride, stride * 2);

	int num_half_points = (int)even_parts.size();
	// Forward:
	//   A_even(X²) + X * A_odd(X²)
	// = A_even(X²) + e^(-2pi * i * k/N) * A_odd(X²)
	// Inverse:
	//   A_even(X²) + X * A_odd(X²)
	// = A_even(X²) + e^(2pi * i * k/N) * A_odd(X²)
	const complex prefactor = (inverse ? (2i*PI) : (-2i*PI)) / (double)(num_half_points * 2);

	std::vector<complex> evaluated_points(num_half_points * 2);
	for (int k = 0; k < num_half_points; k++) {
		complex p = even_parts[k];
		complex q = odd_parts[k] * exp(prefactor * (double)k);
		evaluated_points[k] = p + q;
		evaluated_points[k + num_half_points] = p - q;
	}

	if (points == (int)signal.size() && inverse) {
		// On the highest level of the recursion, divide by N for the required normalization
		// if this is an inverse transform.
		// (Vandermonde matrix for inverse fourier transformation has entries e^(2i*pi*n*k/N) / N.
		//  Division by N has to occur at the highest level of the recursion, i.e. here.)
		for (int i = 0; i < points; i++) {
			evaluated_points[i] /= points;
		}
	}
	return evaluated_points;
}


// Wrapper for the recursive implementation of the FFT.
std::vector<complex> evaluate_polynomial(const std::vector<double>& polynomial, int num_samples) {
	// Pad coefficient vector with zeros to power of two length.
	std::vector<complex> padded_polynomial;
	for (int i = 0; i < num_samples; i++) {
		if (i < (int)polynomial.size())
			padded_polynomial.push_back(polynomial[i]);
		else
			padded_polynomial.push_back(0);
	}
	return _fft_impl(padded_polynomial, false, num_samples, 0, 1);
}


// Wrapper for he recursive implementation of the inverse FFT.
std::vector<double> recover_coefficients(const std::vector<complex>& value_representation) {
	int num_support_points = (int)value_representation.size();
	std::vector<complex> complex_result = _fft_impl(value_representation, true, num_support_points, 0, 1);
	std::vector<double> real_result;
	for (int i = 0; i < (int)complex_result.size(); i++) {
		// Keep only the real part, imaginary part should be zero.
		real_result.push_back(complex_result[i].real());
	}
	return real_result;
}


/* 
* FFT-based n * log(n) polynomial multiplication algorithm
* 
* Polynomials are given as coefficients with the highest order coefficient at the end
* of the vectors. Polynomials may safely be padded with zeros to the right.
* Return value is a coefficient representation of the product polynomial truncated to
* the highest order non-zero coefficient.
*/
std::vector<double> fast_polynomial_multiplication(const std::vector<double>& p1, const std::vector<double>& p2) {
	// Number of supporting (x, value) pairs to uniquely define the product polynomial
	// (equal to the maximum degree of the product plus one).
	int required_support = (int)p1.size() + (int)p2.size() - 1;
	// Support that will be used instead, at least 'required_support'.
	int support = 1 << (int)ceil(log2((double)required_support));

	// Transform polynomial representation to (x, value) pair representation using discrete fourier transform.
	std::vector<complex> value_repr1 = evaluate_polynomial(p1, support);
	std::vector<complex> value_repr2 = evaluate_polynomial(p2, support);

	// Multiply polynomials in the (x, value) representation.
	for (int i = 0; i < (int)value_repr1.size(); i++) {
		value_repr1[i] *= value_repr2[i];
	}

	// Transform result back to coefficient representation.
	std::vector<double> product = recover_coefficients(value_repr1);

	// Remove highest order terms that are zero.
	int i = (int)product.size() - 1;
	while (i > 1 && std::abs((double)product[i]) < 1e-10) i--;
	product.erase(product.begin() + i + 1, product.end());
	return product;
}


// Prints the given polynomial to the console in one line.
// Assumes cout has fixed precision flag already set.
void print_polynomial(const std::vector<double>& p) {
	std::cout << p[0];
	for (int i = 1; i < (int)p.size(); i++) {
		std::cout << " " << p[i];
	}
	std::cout << std::endl;
}


// Testing code
int main() {
	std::cout << std::fixed;

	int samples = 8;
	// From lowest order coefficient to the highest, index gives exponent
	const std::vector<double> polynomial1 = { 1, 1, 0, 0, 1, 1, 0, 1 };
	const std::vector<double> polynomial2 = { 0, 1, std::sqrt(2), 1, 1, 0, 1, 1};
	const std::vector<double> polynomial3 = { 0, PI, 1, 5 };

	auto true_result = test_multiply(polynomial3, polynomial2);
	auto fpm_result = fast_polynomial_multiplication(polynomial3, polynomial2);

	std::cout << "Correct result:" << std::endl;
	print_polynomial(true_result);
	std::cout << "Result from fast multiplication implementation:" << std::endl;
	print_polynomial(fpm_result);

	return 0;
}
