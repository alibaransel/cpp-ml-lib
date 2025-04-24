#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;
using std::cout;

template <typename T>
using matrix = vector<vector<T>>;
template <typename T>
using tensor = vector<vector<vector<T>>>;

matrix<double> matrixMultiplication(matrix<double> a, matrix<double> b) {
    matrix<double> c(a.size(), vector<double>(b[0].size()));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < b[0].size(); j++) {
            for (int k = 0; k < a[0].size(); k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}

double dotProduct(vector<double> a, vector<double> b) {
    double c = 0;
    for (int i = 0; i < a.size(); i++) {
        c += a[i] * b[i];
    }
    return c;
}

void printMatrix(matrix<double> a) {
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            cout << a[i][j] << ' ';
        }
        cout << endl;
    }
}

matrix<double> transpose(matrix<double> a) {
    matrix<double> aT = matrix<double>(a[0].size(), vector<double>(a.size()));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            aT[j][i] = a[i][j];
        }
    }
    return aT;
}

double sigmoid(double z) {
    return 1 / (1 + exp(-z));
}

double sigmoidDerivative(double a) {
    return a * (1 - a);
}

matrix<double> matrixForEach(matrix<double> a, double f(double)) {
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            a[i][j] = f(a[i][j]);
        }
    }
    return a;
}

matrix<double> matrixAndExpandedVectorAddition(matrix<double> a, vector<double> v) {
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            a[i][j] += v[i];
        }
    }
    return a;
}

vector<double> vectorForEach(vector<double> v, double f(double)) {
    for (int i = 0; i < v.size(); i++) {
        v[i] = f(v[i]);
    }
    return v;
}

matrix<double> matrixSubtraction(matrix<double> a, matrix<double> b) {
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            a[i][j] -= b[i][j];
        }
    }
    return a;
}

vector<double> matrixSumOnDim2(matrix<double> a) {
    vector<double> s(a.size(), 0);
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            s[i] += a[i][j];
        }
    }
    return s;
}

matrix<double> matrixSingleProduct(matrix<double> a, matrix<double> b) {
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            a[i][j] *= b[i][j];
        }
    }
    return a;
}

matrix<double> matrixScalarProduct(double k, matrix<double> a) {
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            a[i][j] *= k;
        }
    }
    return a;
}

vector<double> vectorScalarProduct(double k, vector<double> v) {
    for (int i = 0; i < v.size(); i++) {
        v[i] *= k;
    }
    return v;
}

vector<double> vectorSubtraction(vector<double> a, vector<double> b) {
    for (int i = 0; i < a.size(); i++) {
        a[i] -= b[i];
    }
    return a;
}

matrix<double> matrixAddition(matrix<double> a, matrix<double> b) {
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            a[i][j] += b[i][j];
        }
    }
    return a;
}

matrix<double> matrixSingleAddition(matrix<double> a, double k) {
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            a[i][j] += k;
        }
    }
    return a;
}

double matrixSum(matrix<double> a) {
    double s = 0;  // Improve later (max bound error may occur)
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            s += a[i][j];
        }
    }
    return s;
}

double matrixMean(matrix<double> a) {
    double s = matrixSum(a);
    s /= a.size();
    s /= a[0].size();
    return s;
}

bool threshold(double x) {
    return x >= 0.5;
}

matrix<bool> matrixForEachDoubleToBool(matrix<double> a, bool f(double)) {
    matrix<bool> b(a.size(), vector<bool>(a[0].size()));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            b[i][j] = f(a[i][j]);
        }
    }
    return b;
}

matrix<bool> matrixLogicXOR(matrix<bool> a, matrix<bool> b) {
    matrix<bool> c(a.size(), vector<bool>(a[0].size()));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            c[i][j] = a[i][j] xor b[i][j];
        }
    }
    return c;
}

matrix<double> matrixForEachBoolToDouble(matrix<bool> a, double f(bool)) {
    matrix<double> b(a.size(), vector<double>(a[0].size()));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            b[i][j] = f(a[i][j]);
        }
    }
    return b;
}

double boolToDouble(bool x) {
    if (x) return 1;
    return 0;
}

class ANNVec {
   private:
    vector<int> stucture;
    int layerCount;
    tensor<double> w;
    matrix<double> b;

   public:
    ANNVec(vector<int> structure) {  // TODO: separate input size, maybe separate crete and initialize
        this->stucture = structure;
        this->layerCount = structure.size();
        w = tensor<double>(structure.size() - 1);
        b = matrix<double>(structure.size() - 1);
        for (int iL = 1; iL < layerCount; iL++) {
            w[iL - 1] = matrix<double>(structure[iL], vector<double>(structure[iL - 1], 0.5));
            b[iL - 1] = vector<double>(structure[iL], 0.5);
        }
        cout << fixed << setprecision(7);
    };
    ~ANNVec() {};

    void train(matrix<double> xTrain, matrix<double> yTrain, matrix<double> xTest, matrix<double> yTest, double learningRate, int epochs) {
        tensor<double> z(layerCount);
        tensor<double> a(layerCount);
        matrix<double> output;
        double loss;
        vector<double> epochLossGraph(epochs);
        tensor<double> delta(layerCount);
        tensor<double> gradW(layerCount);
        matrix<double> gradB(layerCount);

        auto trainStartTime = chrono::high_resolution_clock::now();
        for (int iE = 0; iE < epochs; iE++) {
            z[0] = transpose(xTrain);
            a[0] = transpose(xTrain);
            for (int iL = 1; iL < layerCount; iL++) {
                z[iL] = matrixAndExpandedVectorAddition(matrixMultiplication(w[iL - 1], a[iL - 1]), b[iL - 1]);
                a[iL] = matrixForEach(z[iL], sigmoid);
            }

            output = a.back();
            loss = -matrixMean(
                matrixAddition(
                    matrixSingleProduct(transpose(yTrain), matrixForEach(output, log)),
                    matrixSingleProduct(matrixScalarProduct(-1, matrixSingleAddition(transpose(yTrain), -1)), matrixForEach(matrixScalarProduct(-1, matrixSingleAddition(output, -1)), log))));
            epochLossGraph[iE] = loss;

            delta = {};
            gradW = {};
            gradB = {};
            delta[layerCount - 1] = matrixSingleProduct(matrixSubtraction(output, transpose(yTrain)), matrixForEach(output, sigmoidDerivative));  // Inner of cross

            for (int iL = layerCount - 2; iL > -1; iL--) {
                gradW[iL] = matrixMultiplication(delta[iL + 1], transpose(a[iL]));
                gradB[iL] = matrixSumOnDim2(delta[iL + 1]);

                if (iL > 0) {
                    delta[iL] = matrixSingleProduct(matrixMultiplication(transpose(w[iL]), delta[iL + 1]), matrixForEach(a[iL], sigmoidDerivative));  // Inner of cross
                }
            }

            for (int iL = 0; iL < layerCount - 1; iL++) {
                w[iL] = matrixSubtraction(w[iL], matrixScalarProduct(learningRate, gradW[iL]));
                b[iL] = vectorSubtraction(b[iL], vectorScalarProduct(learningRate, gradB[iL]));
            }

            if (true || (iE + 1) % (epochs / 100 + 1) == 0) {  // Temporary always true
                cout << "Epoch %" << 100.0 * (iE + 1) / epochs << ", Loss: " << loss << endl;
            }
        }
        auto trainEndTime = chrono::high_resolution_clock::now();
        auto trainDuration = chrono::duration_cast<chrono::milliseconds>(trainEndTime - trainStartTime);

        tensor<double> zTest(layerCount);
        tensor<double> aTest(layerCount);

        zTest[0] = transpose(xTest);
        aTest[0] = transpose(xTest);

        for (int iL = 1; iL < layerCount; iL++) {
            zTest[iL] = matrixAndExpandedVectorAddition(matrixMultiplication(w[iL - 1], aTest[iL - 1]), b[iL - 1]);
            aTest[iL] = matrixForEach(zTest[iL], sigmoid);
        }

        matrix<double> outputTest = aTest.back();
        double testLoss = -matrixMean(
            matrixAddition(
                matrixSingleProduct(transpose(yTest), matrixForEach(outputTest, log)),
                matrixSingleProduct(matrixScalarProduct(-1, matrixSingleAddition(transpose(yTest), -1)), matrixForEach(matrixScalarProduct(-1, matrixSingleAddition(outputTest, -1)), log))));
        cout << "Test Loss: " << testLoss << endl;

        matrix<bool> r1 = matrixForEachDoubleToBool(transpose(outputTest), threshold);
        matrix<bool> r2 = matrixForEachDoubleToBool(yTest, threshold);
        double predTrue = r1.size() - matrixSum(matrixForEachBoolToDouble(matrixLogicXOR(r1, r2), boolToDouble));
        double accuracy = predTrue / r1.size();
        cout << "Train Duration (ms): " << trainDuration.count() << endl;
        cout << "Accuracy: " << accuracy << endl;
    }
};

tuple<matrix<double>, matrix<double>> getDataset() {
    ifstream file("Student_Performance.csv");
    if (!file.is_open())
        exit(0);  // improve later
    string line;
    getline(file, line);
    istringstream lineSS(line);
    vector<string> keys;
    string key;
    while (getline(lineSS, key, ','))
        keys.push_back(key);

    matrix<double> X;
    matrix<double> y;  // TODO: Make y 2d
    vector<double> lineData;
    string stringData;
    int n = 10000;
    while (n > 0 && getline(file, line)) {  // TODO: decrease n for each iteration
        istringstream newLineSS(line);
        getline(newLineSS, stringData, ',');
        lineData.push_back(stod(stringData) / 10);
        getline(newLineSS, stringData, ',');
        lineData.push_back(stod(stringData) / 100);
        getline(newLineSS, stringData, ',');
        double a = stringData == "Yes" ? 1.0 : 0.0;
        lineData.push_back(a);
        getline(newLineSS, stringData, ',');
        lineData.push_back(stod(stringData) / 10);
        getline(newLineSS, stringData, ',');
        lineData.push_back(stod(stringData) / 10);
        getline(newLineSS, stringData, ',');
        y.push_back({stod(stringData) / 100});
        X.push_back(lineData);
        lineData.clear();
    }
    file.close();
    return {X, y};
}

int main() {
    /*
    matrix<double>xTrain = {
        {0.1, 0.2},
        {0.3, 0.4},
    };
    matrix<double>yTrain = {
        {0.4},
        {0.6},
    };
    matrix<double>xTest = {
        {0.1, 0.2},
        {0.3, 0.4},
    };
    matrix<double>yTest = {
        {0.4},
        {0.6},
    };
    */

    matrix<double> xData, yData, xTrain, yTrain, xTest, yTest;
    tie(xData, yData) = getDataset();
    size_t i;
    for (i = 0; i < xData.size() * 0.9; i++) {
        xTrain.push_back(xData[i]);
        yTrain.push_back(yData[i]);
    }
    for (; i < xData.size(); i++) {
        xTest.push_back(xData[i]);
        yTest.push_back(yData[i]);
    }

    int inputSize = xTrain[0].size();
    vector<int> structure = {inputSize, 3, 1};

    ANNVec model = ANNVec(structure);

    double learningRate = 1e-4;
    int epochs = 1e4;

    model.train(xTrain, yTrain, xTest, yTest, learningRate, epochs);

    return 0;
}
