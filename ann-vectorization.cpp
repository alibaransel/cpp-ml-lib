#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;
using std::cout;

vector<vector<double>> matrixMultiplication(vector<vector<double>> a, vector<vector<double>> b) {
    vector<vector<double>> c(a.size(), vector<double>(b[0].size()));
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

void printMatrix(vector<vector<double>> a) {
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            cout << a[i][j] << ' ';
        }
        cout << endl;
    }
}

vector<vector<double>> transpose(vector<vector<double>> a) {
    vector<vector<double>> aT = vector<vector<double>>(a[0].size(), vector<double>(a.size()));
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

vector<vector<double>> matrixForEach(vector<vector<double>> a, double f(double)) {
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            a[i][j] = f(a[i][j]);
        }
    }
    return a;
}

vector<vector<double>> matrixAndExpandedVectorAddition(vector<vector<double>> a, vector<double> v) {
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

vector<vector<double>> matrixSubtraction(vector<vector<double>> a, vector<vector<double>> b) {
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            a[i][j] -= b[i][j];
        }
    }
    return a;
}

vector<double> matrixSumOnDim2(vector<vector<double>> a) {
    vector<double> s(a.size(), 0);
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            s[i] += a[i][j];
        }
    }
    return s;
}

vector<vector<double>> matrixSingleProduct(vector<vector<double>> a, vector<vector<double>> b) {
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            a[i][j] *= b[i][j];
        }
    }
    return a;
}

vector<vector<double>> matrixScalarProduct(double k, vector<vector<double>> a) {
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

vector<vector<double>> matrixAddition(vector<vector<double>> a, vector<vector<double>> b) {
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            a[i][j] += b[i][j];
        }
    }
    return a;
}

vector<vector<double>> matrixSingleAddition(vector<vector<double>> a, double k) {
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            a[i][j] += k;
        }
    }
    return a;
}

double matrixSum(vector<vector<double>> a) {
    double s = 0;  // Improve later (max bound error may occur)
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            s += a[i][j];
        }
    }
    return s;
}

double matrixMean(vector<vector<double>> a) {
    double s = matrixSum(a);
    s /= a.size();
    s /= a[0].size();
    return s;
}

bool threshold(double x) {
    return x >= 0.5;
}

vector<vector<bool>> matrixForEachDoubleToBool(vector<vector<double>> a, bool f(double)) {
    vector<vector<bool>> b(a.size(), vector<bool>(a[0].size()));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            b[i][j] = f(a[i][j]);
        }
    }
    return b;
}

vector<vector<bool>> matrixLogicXOR(vector<vector<bool>> a, vector<vector<bool>> b) {
    vector<vector<bool>> c(a.size(), vector<bool>(a[0].size()));
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            c[i][j] = a[i][j] xor b[i][j];
        }
    }
    return c;
}

vector<vector<double>> matrixForEachBoolToDouble(vector<vector<bool>> a, double f(bool)) {
    vector<vector<double>> b(a.size(), vector<double>(a[0].size()));
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

int main() {
    vector<vector<double>> xTrain = {
        {0.1, 0.2},
        {0.3, 0.4},
    };
    vector<vector<double>> yTrain = {
        {0.4},
        {0.6},
    };

    int inputSize = xTrain[0].size();
    vector<int> structure = {inputSize, 3, 1};
    int layerCount = structure.size();

    vector<vector<vector<double>>> w(structure.size() - 1);
    vector<vector<double>> b(structure.size() - 1);

    for (int iL = 1; iL < layerCount; iL++) {
        w[iL - 1] = vector<vector<double>>(structure[iL], vector<double>(structure[iL - 1], 0.5));
        b[iL - 1] = vector<double>(structure[iL], 0.5);
    }

    double learningRate = 1e-4;
    int epochs = 1e5;

    vector<vector<vector<double>>> z(layerCount);
    vector<vector<vector<double>>> a(layerCount);
    vector<vector<double>> output;
    double loss;
    vector<double> epochLossGraph(epochs);
    vector<vector<vector<double>>> delta(layerCount);
    vector<vector<vector<double>>> gradW(layerCount);
    vector<vector<double>> gradB(layerCount);

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

        if (true || iE % 100 == 0) {  // Temporary always true
            cout << "Epoch " << iE << ", Loss: " << loss << endl;
        }
    }
    auto trainEndTime = chrono::high_resolution_clock::now();
    auto trainDuration = chrono::duration_cast<chrono::milliseconds>(trainEndTime - trainStartTime);

    vector<vector<double>> xTest = {
        {0.1, 0.2},
        {0.3, 0.4},
    };
    vector<vector<double>> yTest = {
        {0.4},
        {0.6},
    };

    vector<vector<vector<double>>> zTest(layerCount);
    vector<vector<vector<double>>> aTest(layerCount);

    zTest[0] = transpose(xTest);
    aTest[0] = transpose(xTest);

    for (int iL = 1; iL < layerCount; iL++) {
        zTest[iL] = matrixAndExpandedVectorAddition(matrixMultiplication(w[iL - 1], aTest[iL - 1]), b[iL - 1]);
        aTest[iL] = matrixForEach(zTest[iL], sigmoid);
    }

    vector<vector<double>> outputTest = aTest.back();
    double testLoss = -matrixMean(
        matrixAddition(
            matrixSingleProduct(transpose(yTest), matrixForEach(outputTest, log)),
            matrixSingleProduct(matrixScalarProduct(-1, matrixSingleAddition(transpose(yTest), -1)), matrixForEach(matrixScalarProduct(-1, matrixSingleAddition(outputTest, -1)), log))));
    cout << "Test Loss: " << testLoss << endl;

    vector<vector<bool>> r1 = matrixForEachDoubleToBool(transpose(outputTest), threshold);
    vector<vector<bool>> r2 = matrixForEachDoubleToBool(yTest, threshold);
    double predTrue = r1.size() - matrixSum(matrixForEachBoolToDouble(matrixLogicXOR(r1, r2), boolToDouble));
    double accuracy = predTrue / r1.size();
    cout << "Train Duration (ms): " << trainDuration.count() << endl;
    cout << "Accuracy: " << accuracy << endl;

    return 0;
}
