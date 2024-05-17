
#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include<future>
#include <chrono>
#include <random>

using namespace std;
using namespace std::chrono;

template<typename T>
class Matrix
{
public:
	T** A;//матрица
	int m; //количество строк
	int n;//количество столбцов
	//конструктор
	Matrix()
	{
		m = 0;
		n = 0;
		A = nullptr;
	}
	Matrix(int x, int y) : m(x), n(y)
	{
		A = new T * [m];
		for (int i = 0; i < m; i++) {
			A[i] = new T[n];
		}
	}
	Matrix(const Matrix& matr)
	{
		m = matr.m;
		n = matr.n;
		A = new T * [m];
		for (int i = 0; i < m; i++) {
			A[i] = new T[n];
			for (int j = 0; j < n; j++) {
				A[i][j] = matr.A[i][j];
			}
		}
	}
	
	friend ostream& operator<<(ostream& os, const Matrix& matrix) {
		for (int i = 0; i < matrix.m; ++i) {
			for (int j = 0; j < matrix.n; ++j)
				os << matrix.A[i][j] << " ";
			os << endl;
		}
		return os;
	}

	friend istream& operator>>(istream& is, Matrix& matrix) {
		cout << "enter number of rows and columns of the matrix:\n";
		is >> matrix.m >> matrix.n;
		matrix.A = new T * [matrix.m];
		for (int i = 0; i < matrix.m; ++i) {
			matrix.A[i] = new T[matrix.n];
			for (int j = 0; j < matrix.n; ++j)
			{
				cout << "Enter the number: row - " << i + 1 << "; column - " << j + 1 << ": ";
				is >> matrix.A[i][j];
			}
		}
		return is;
	}

	//перегрузка оператора +
	Matrix operator + (const Matrix& matr) const
	{
		if (m != matr.m || n != matr.n)
		{
			cerr << "impossible to add!\n";
			return Matrix();
		}
		Matrix result(m, n);
		// Запуск параллельных потоков для сложения матриц
		/*Этот цикл создает потоки и добавляет их в вектор threads.
		Каждый поток выполняет лямбда-функцию, которая вычисляет результат сложения двух матриц, используя переменную i. 
		Переменная i используется для определения строки, для которой производится сложение.
		Лямбда-функция получает четыре ссылки:
		* this - ссылка на текущий объект
		* result - ссылка на результат
		* matr - ссылка на другую матрицу
		* i - номер строки*/
		vector<thread> threads;
		for (int i = 0; i < m; i ++)
		{
			threads.push_back(thread([this, &result, &matr, i]() 
			{
				for (int j = 0; j < n; j ++)
				{
					result.A[i][j] = A[i][j] + matr.A[i][j];
				}
			}));
		}
		// Ожидание завершения всех потоков. Этот цикл ожидает завершения всех потоков в векторе threads
		for (thread& thread : threads) 
		{
			thread.join();
		}
		//После завершения всех потоков функция возвращает матрицу result, которая содержит результат сложения двух матриц.
		return result;
	}
	 Matrix addWithAsync (const Matrix& matr, int block_size) const
	{
		if (m != matr.m || n != matr.n)
		{
			cerr << "impossible to add!n";
			return Matrix();
		}
		Matrix result(m, n);
		vector<future<void>> futures;
		for (int i = 0; i < m; i += block_size) // Внешний цикл по i и внутренний цикл по j делят матрицы на блоки
		{
			for (int j = 0; j < n; j += block_size)
			{
				futures.push_back(async(launch::async, [&, i, j]() //Внутри каждого блока запускается асинхронная задача 
				{
						for (int ii = i; ii < min(i + block_size, m); ii++)
						{
							for (int jj = j; jj < min(j + block_size, n); jj++)
							{
								result.A[ii][jj] = A[ii][jj] + matr.A[ii][jj];
							}
						}
				}));
			}
		}
		
		/*После создания потоков в цикле ожидания основной поток дожидается завершения всех потоков. 
		Функция f.get() используется для получения результатов вычислений из потоков.*/
		for (auto& f : futures)
		{
			f.get();
		}
		return result;
	 }
	//перегрузка оператора -
	Matrix operator - (const Matrix& matr) const
	{
		if (m != matr.m || n != matr.n)
		{
			cerr << "impossible to substract!\n";
			return Matrix();
		}
		Matrix result(m, n);
		vector<thread> threads;
		for (int i = 0; i < m; i++)
		{
			threads.push_back(thread([this, &result, &matr, i]()
			{
					for (int j = 0; j < n; j++)
					{
						result.A[i][j] = A[i][j] - matr.A[i][j];
					}
			}));
		}
		for (thread& thread : threads)
		{
			thread.join();
		}
		return result;
	}
	Matrix subWithAsync(const Matrix& matr, int block_size) const
	{
		if (m != matr.m || n != matr.n)
		{
			cerr << "impossible to substract!\n";
			return Matrix();
		}
		Matrix result(m, n);
		vector<future<void>> futures;
		for (int i = 0; i < m; i += block_size)
		{
			for (int j = 0; j < n; j += block_size)
			{
				futures.push_back(async(launch::async, [&, i, j]()
				{
						for (int ii = i; ii < min(i + block_size, m); ii++)
						{
							for (int jj = j; jj < min(j + block_size, n); jj++)
							{
								result.A[ii][jj] = A[ii][jj] - matr.A[ii][jj];
							}
						}
				}));
			}
		}
		for (auto& f : futures)
		{
			f.get();
		}
		return result;
	}
	//перегрузка оператора умножения на скаляр
	Matrix operator * (const T& scalar) const
	{
		Matrix result(m, n);
		vector<thread> threads;
		for (int i = 0; i < m; i++)
		{
			threads.push_back(thread([this, &result, scalar, i]()
			{
				for (int j = 0; j < n; j++)
				{
					result.A[i][j] = A[i][j] * scalar;
				}
			}));
		}
		for (thread& thread : threads)
		{
			thread.join();
		}
		return result;
	}
	Matrix scalMultWithAsync(const T& scalar, int block_size) const
	{
		Matrix result(m, n);
		vector<future<void>> futures;
		for (int i = 0; i < m; i += block_size)
		{
			for (int j = 0; j < n; j += block_size)
			{
				futures.push_back(async(launch::async, [&, i, j]()
				{
						for (int ii = i; ii < min(i + block_size, m); ii++)
						{
							for (int jj = j; jj < min(j + block_size, n); jj++)
							{
								result.A[ii][jj] = A[ii][jj] * scalar;
							}
						}
				}));
			}
		}
		for (auto& f : futures)
		{
			f.get();
		}
		return result;
	}
	//перегрузка оператора умножения
	Matrix operator*(const Matrix& matr) const {
		if (n != matr.m)
		{
			cerr << "impossible to multiply!\n";
			return Matrix();
		}

		Matrix result(m, matr.n);
		vector<thread> threads;

		for (int i = 0; i < m; i++) {
			threads.push_back(thread([this, &result, &matr, i]() {
				for (int j = 0; j < matr.n; ++j) {
					result.A[i][j] = 0;
					for (int k = 0; k < n; ++k) {
						result.A[i][j] += A[i][k] * matr.A[k][j];
					}
				}
			}));
		}

		for (thread& thread : threads)
		{
			thread.join();
		}
		return result;
	}
	Matrix multWithAsync(const Matrix& matr, int block_size) const {
		if (n != matr.m)
		{
			cerr << "impossible to multiply!\n";
			return Matrix();
		}

		Matrix result(m, matr.n);
		vector<future<void>> futures;
		for (int i = 0; i < m; i += block_size)
		{
			for (int j = 0; j < matr.n; j += block_size)
			{
				futures.push_back(async(launch::async, [&, i, j]()
				{
						for (int ii = i; ii < min(i + block_size, m); ii++)
						{
							for (int jj = j; jj < min(j + block_size, matr.n); jj++)
							{
								result.A[ii][jj] = 0;
								for (int k = 0; k < n; k++)
								{
									result.A[ii][jj] += A[ii][k] * matr.A[k][jj];
								}
							}
						}
				}));
			}
		}
		for (auto& f : futures)
		{
			f.get();
		}
		return result;
	}

	double determinant() {
		if (m != n) {
			throw "Unable to calculate determinant";
			return 0;
		}
		if (m == 1) {
			return A[0][0];
		}

		if (m == 2) {
			return A[0][0] * A[1][1] - A[0][1] * A[1][0];
		}
		double det = 0;
		vector<thread> threads; //Вектор threads используется для хранения потоков, которые будут выполнять параллельные вычисления

		for (int j = 0; j < n; j++) { //Для каждого столбца j матрицы создается отдельный поток
			threads.push_back(thread([&, j]()
				{
					Matrix minor(m - 1, n - 1);
					for (int i = 1; i < m; i++) {
						int k = 0;
						for (int l = 0; l < n; l++) {
							if (l != j) {
								minor.A[i - 1][k] = A[i][l];
								k++;
							}
						}
					}
					det += pow(-1, j) * A[0][j] * minor.determinant();
				}));
		}

		for (auto& thread : threads) {
			thread.join();
		}

		return det;
	}
	
	double determinantWithAsync(int block_size) {
		if (m != n) {
			throw "Unable to calculate determinant";
			return 0;
		}
		if (m == 1) {
			return A[0][0];
		}

		if (m == 2) {
			return A[0][0] * A[1][1] - A[0][1] * A[1][0];
		}

		double det = 0;
		vector<future<double>> futures;

		for (int j = 0; j < n; j += block_size) {
			futures.push_back(async(launch::async, [&, j]() {
				double block_det = 0;
				for (int jj = j; jj < min(j + block_size, n); jj++) {
					Matrix minor(m - 1, n - 1);
					for (int i = 1; i < m; i++) {
						int k = 0;
						for (int l = 0; l < n; l++) {
							if (l != jj) {
								minor.A[i - 1][k] = A[i][l];
								k++;
							}
						}
					}
					block_det += pow(-1, jj) * A[0][jj] * minor.determinantWithAsync(block_size); //Каждая задача вычисляет определитель для своего блока и добавляет результат в block_det.
				}
				return block_det;
			}));
		}

		for (auto& future : futures) {
			det += future.get();
		}

		return det;
	}

	double cofactor(int row, int column)
	{
		Matrix minor(m - 1, n - 1);
		int m1, n1, sign;
		m1 = 0;

		vector<thread> threads;

		for (int i = 0; i < m - 1; i++) 
		{
			threads.emplace_back([&, i] 
				{
				if (i == row - 1) 
				{
					m1 = 1;
				}
				n1 = 0;
				for (int j = 0; j < m - 1; j++) 
				{
					if (j == column - 1) 
					{
						n1 = 1;
					}
					minor.A[i][j] = A[i + m1][j + n1];
				}
			});
		}

		for (auto& thread : threads) 
		{
			thread.join();
		}

		if ((row + column) % 2 == 0) 
		{
			sign = 1;
		}
		else 
		{
			sign = -1;
		}

		return sign * minor.determinant();
	}
	double cofactorWithAsync(int row, int column, int block_size)
	{
		Matrix minor(m - 1, n - 1);
		int m1, n1, sign;
		m1 = 0;

		vector<future<void>> futures;

		for (int i = 0; i < m - 1; i += block_size)
		{
			futures.push_back(async(launch::async, [&, i]()
			{
					for (int k = i; k < min(i + block_size, m - 1); k++)
					{
						if (k == row - 1)
						{
							m1 = 1;
						}
						n1 = 0;

						for (int j = 0; j < n - 1; j++)
						{
							if (j == column - 1)
							{
								n1 = 1;
							}
							minor.A[k - i][j] = A[k + m1][j + n1];
						}
					}
			}));
		}

		for (auto& future : futures)
		{
			future.wait();
		}

		if ((row + column) % 2 == 0)
		{
			sign = 1;
		}
		else
		{
			sign = -1;
		}

		return sign * minor.determinantWithAsync(block_size);
	}
	Matrix cofactorMatrix()
	{
		Matrix copy(m, n);
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				copy.A[i][j] = A[i][j];
			}
		}
		Matrix cofactorM(m, n);

		vector<thread> threads;

		for (int i = 0; i < m; i++) 
		{
			for (int j = 0; j < n; j++) 
			{
				threads.emplace_back([&, i, j]
				{
					cofactorM.A[i][j] = copy.cofactor(i + 1, j + 1);
				});
			}
		}

		for (auto& thread : threads) 
		{
			thread.join();
		}

		return cofactorM;
	}
	
	Matrix cofactorMatrixWithAsync(int block_size) {
		Matrix copy(m, n);
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				copy.A[i][j] = A[i][j];
			}
		}
		Matrix cofactorM(m, n);

		vector<future<void>> futures;

		for (int i = 0; i < m; i += block_size) {
			for (int j = 0; j < n; j += block_size) {
				futures.push_back(async(launch::async, [&, i, j]() {
					for (int ii = i; ii < min(i + block_size, m); ii++) {
						for (int jj = j; jj < min(j + block_size, n); jj++) {
							cofactorM.A[ii][jj] = copy.cofactorWithAsync(ii + 1, jj + 1, block_size);
						}
					}
					}));
			}
		}

		for (auto& future : futures) {
			future.get();
		}

		return cofactorM;
	}

	Matrix transposeMatrix()
	{
		Matrix transpose(n, m);

		vector<thread> threads;
		for (int i = 0; i < n; ++i) 
		{
			threads.push_back(thread([&,i]() 
			{
				for (int j = 0; j < m; ++j) 
				{
					transpose.A[i][j] = A[j][i];
				}
			}));
		}
		for (thread& thread : threads) 
		{
			thread.join();
		}
		return transpose;
	}
	Matrix transposeMatrixWithAsync(int block_size) {
		Matrix transpose(n, m);

		vector<future<void>> futures;

		for (int i = 0; i < m; i += block_size) {
			for (int j = 0; j < n; j += block_size) {
				futures.push_back(async(launch::async, [&, i, j]() {
					for (int ii = i; ii < min(i + block_size, m); ii++) {
						for (int jj = j; jj < min(j + block_size, n); jj++) {
							transpose.A[jj][ii] = A[ii][jj];
						}
					}
					}));
			}
		}

		for (auto& future : futures) {
			future.get();
		}

		return transpose;
	}

	Matrix operator !()
	{
		Matrix copy(m, n);
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				copy.A[i][j] = A[i][j];
			}
		}
		double det = copy.determinant();
		if ((m != n) || (det == 0)) {
			throw "Unable to find the inverse matrix";
		}
		else {
			Matrix cofactorMat = copy.cofactorMatrix();
			Matrix adjointMat = cofactorMat.transposeMatrix();
			Matrix inverse = adjointMat * (1 / det);
			return inverse;
		}
	}

	Matrix inverseWithAsync(int block_size)
	{
		Matrix copy(m, n);
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				copy.A[i][j] = A[i][j];
			}
		}
		double det = copy.determinantWithAsync(block_size);
		if ((m != n) || (det == 0)) {
			throw "Unable to find the inverse matrix";
		}
		else {
			Matrix cofactorMat = copy.cofactorMatrixWithAsync(block_size);
			Matrix adjointMat = cofactorMat.transposeMatrixWithAsync(block_size);
			Matrix inverse = adjointMat.scalMultWithAsync(1 / det, block_size);
			return inverse;
		}
	}


	~Matrix() {
		if (A != nullptr)
		{
			for (int i = 0; i < m; i++) {
				delete[] A[i];
			}
			delete[] A;
		}
	}
};

int main()
{
	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<int> dist(0, 100);
	//16.1
	//int blocks = 1;
		/*for (int m = 0; m <= 2400; m += 60)
		{

			Matrix<double> A(m, m), B(m, m);

			for (int i = 0; i < m; ++i) {
				for (int j = 0; j < m; ++j) {
					A.A[i][j] = dist(gen);
				}
			}

			for (int i = 0; i < m; ++i) {
				for (int j = 0; j < m; ++j) {
					B.A[i][j] = dist(gen);
				}
			}
			auto start = high_resolution_clock::now();
			Matrix<double> E = A.addWithAsync(B, blocks);
			auto stop = high_resolution_clock::now();

			auto duration = duration_cast<milliseconds>(stop - start);
			cout << " Size: " << m << " Addition time: " << duration.count() << " ms\n";
			blocks += 30;
		}*/

		/*for (int m = 0; m <= 2400; m += 60)
		{
			Matrix<double> A(m, m), B(m, m);

			for (int i = 0; i < m; ++i) {
				for (int j = 0; j < m; ++j) {
					A.A[i][j] = dist(gen);
				}
			}

			for (int i = 0; i < m; ++i) {
				for (int j = 0; j < m; ++j) {
					B.A[i][j] = dist(gen);
				}
			}
			auto start = high_resolution_clock::now();
			Matrix<double> E = A.subWithAsync(B, blocks);
			auto stop = high_resolution_clock::now();

			auto duration = duration_cast<milliseconds>(stop - start);
			cout << " Size: " << m << " Substraction time: " << duration.count() << " ms\n";
			blocks += 30;
		}*/
		/*for (int m = 0; m <= 2400; m += 60)
		{
			Matrix<double> A(m, m);

			for (int i = 0; i < m; ++i) {
				for (int j = 0; j < m; ++j) {
					A.A[i][j] = dist(gen);
				}
			}

			auto start = high_resolution_clock::now();
			Matrix<double> E = A.scalMultWithAsync(3, blocks);
			auto stop = high_resolution_clock::now();

			auto duration = duration_cast<milliseconds>(stop - start);
			cout << " Size: " << m << " Scalar multiplication time: " << duration.count() << " ms\n";
			blocks += 30;
		}*/
		/*for (int m = 0; m <= 1200; m += 60)
		{
			Matrix<double> A(m, m), B(m, m);

			for (int i = 0; i < m; ++i) {
				for (int j = 0; j < m; ++j) {
					A.A[i][j] = dist(gen);
				}
			}

			for (int i = 0; i < m; ++i) {
				for (int j = 0; j < m; ++j) {
					B.A[i][j] = dist(gen);
				}
			}
			auto start = high_resolution_clock::now();
			Matrix<double> E = A.multWithAsync(B, blocks);
			auto stop = high_resolution_clock::now();

			auto duration = duration_cast<milliseconds>(stop - start);
			cout << " Size: " << m << " Multiplication time: " << duration.count() << " ms\n";
			blocks += 30;
		}*/
		/*for (int m = 2; m <= 8; m += 2)
		{
			Matrix<double> A(m, m);

			for (int i = 0; i < m; ++i) {
				for (int j = 0; j < m; ++j) {
					A.A[i][j] = dist(gen);
				}
			}

			auto start = high_resolution_clock::now();
			Matrix<double> E = A.inverseWithAsync(blocks);
			auto stop = high_resolution_clock::now();

			auto duration = duration_cast<milliseconds>(stop - start);
			cout << " Size: " << m << " Inverse matrix calculation time: " << duration.count() << " ms\n";
			blocks += 1; //начальное значение blocks = 1
		}*/
		//16.2
		/*for (int blocks = 100; blocks <= 1000; blocks += 100)
		{
			Matrix<double> A(1000, 1000), B(1000, 1000);

			for (int i = 0; i < 1000; ++i) {
				for (int j = 0; j < 1000; ++j) {
					A.A[i][j] = dist(gen);
				}
			}

			for (int i = 0; i < 1000; ++i) {
				for (int j = 0; j < 1000; ++j) {
					B.A[i][j] = dist(gen);
				}
			}
			auto start = high_resolution_clock::now();
			Matrix<double> E = A.addWithAsync(B, blocks);
			auto stop = high_resolution_clock::now();

			auto duration = duration_cast<milliseconds>(stop - start);
			double threads = (1000 / blocks) * (1000 / blocks);
			cout << "Number of threads: "<< threads << " Addition time: " << duration.count() << " ms\n";
			cout << endl;
		}*/
		/*for (int blocks = 100; blocks <= 1000; blocks += 100)
		{
			Matrix<double> A(1000, 1000), B(1000, 1000);

			for (int i = 0; i < 1000; ++i) {
				for (int j = 0; j < 1000; ++j) {
					A.A[i][j] = dist(gen);
				}
			}

			for (int i = 0; i < 1000; ++i) {
				for (int j = 0; j < 1000; ++j) {
					B.A[i][j] = dist(gen);
				}
			}
			auto start = high_resolution_clock::now();
			Matrix<double> E = A.subWithAsync(B, blocks);
			auto stop = high_resolution_clock::now();

			auto duration = duration_cast<milliseconds>(stop - start);
			double threads = (1000 / blocks) * (1000 / blocks);
			cout << "Number of threads: " << threads << " Substraction time: " << duration.count() << " ms\n";
			cout << endl;
		}*/
		/*for (int blocks = 100; blocks <= 1000; blocks += 100)
		{
			Matrix<double> A(1000, 1000);

			for (int i = 0; i < 1000; ++i) {
				for (int j = 0; j < 1000; ++j) {
					A.A[i][j] = dist(gen);
				}
			}

			auto start = high_resolution_clock::now();
			Matrix<double> E = A.scalMultWithAsync(3, blocks);
			auto stop = high_resolution_clock::now();

			auto duration = duration_cast<milliseconds>(stop - start);
			double threads = (1000 / blocks) * (1000 / blocks);
			cout << "Number of threads: " << threads << " Scalar multiplication time: " << duration.count() << " ms\n";
			cout << endl;
		}*/
		/*for (int blocks = 100; blocks <= 1000; blocks += 100)
		{
			Matrix<double> A(1000, 1000), B(1000, 1000);

			for (int i = 0; i < 1000; ++i) {
				for (int j = 0; j < 1000; ++j) {
					A.A[i][j] = dist(gen);
				}
			}

			for (int i = 0; i < 1000; ++i) {
				for (int j = 0; j < 1000; ++j) {
					B.A[i][j] = dist(gen);
				}
			}
			auto start = high_resolution_clock::now();
			Matrix<double> E = A.multWithAsync(B, blocks);
			auto stop = high_resolution_clock::now();

			auto duration = duration_cast<milliseconds>(stop - start);
			double threads = (1000 / blocks) * (1000 / blocks);
			cout << "Number of threads: " << threads << " Multiplication time: " << duration.count() << " ms\n";
			cout << endl;
		}*/
		/*for (int blocks = 2; blocks <= 6; blocks += 1)
		{
			Matrix<double> A(6, 6);

			for (int i = 0; i < 6; ++i) {
				for (int j = 0; j < 6; ++j) {
					A.A[i][j] = dist(gen);
				}
			}

			auto start = high_resolution_clock::now();
			Matrix<double> E = A.inverseWithAsync(blocks);
			auto stop = high_resolution_clock::now();

			auto duration = duration_cast<milliseconds>(stop - start);
			double threads = (6 / blocks) * (6 / blocks);
			cout << "Number of threads: " << threads << " Inverse matrix calculation time: " << duration.count() << " ms\n";
			cout << endl;
		}*/
		return 0;
}
