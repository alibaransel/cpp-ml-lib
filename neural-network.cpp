#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <tuple>
#include <vector>

using namespace std;

double randomDouble() {
    // return 1.0;
    return (double)rand() / RAND_MAX;
}

class Neuron {
   private:
    vector<double> w;
    double b;

    vector<double> aIn;

    double z;
    double a;
    double delta_z = 0.0;

    double activation(double z) {
        return 1 / (1 + exp(-z));
    }

    double d_activation() {
        return a * (1 - a);
    }

   public:
    Neuron(int inputSize) {
        for (int i = 0; i < inputSize; i++) {
            w.push_back(randomDouble());
        }
        b = randomDouble();
    }

    int getInputSize() {
        return w.size();
    }

    double forward(vector<double> aPreviousLayer) {
        aIn = aPreviousLayer;
        z = b;
        for (int i = 0; i < w.size(); i++) z += w[i] * aIn[i];
        a = activation(z);
        return a;
    }

    void calculateDeltaForOutput(double yII) {
        delta_z = (a - yII) * d_activation();
    }

    vector<double> getWeightDeltaListPart() {
        vector<double> deltaPreviousLayerI;
        for (int w_i = 0; w_i < w.size(); w_i++) deltaPreviousLayerI.push_back(w_i * delta_z);
        return deltaPreviousLayerI;
    }

    void setDelta(double deltaWeightSum) {
        delta_z = deltaWeightSum * d_activation();
    }

    void update(double eta) {
        for (int w_i = 0; w_i < w.size(); w_i++) w[w_i] -= eta * delta_z * aIn[w_i];
        b -= eta * delta_z;
    }
};

class Layer {
   private:
    vector<Neuron> neurons;

    int getInputSize() {
        return neurons[0].getInputSize();
    }

   public:
    Layer(int neuronCount, int inputSize) {
        for (int i = 0; i < neuronCount; i++) neurons.push_back(Neuron(inputSize));
    }

    vector<double> forward(vector<double> x) {
        vector<double> a;
        for (int i = 0; i < neurons.size(); i++) a.push_back(neurons[i].forward(x));
        return a;
    }

    void calculateDeltaForOutput(vector<double> yI) {
        for (int i = 0; i < neurons.size(); i++) neurons[i].calculateDeltaForOutput(yI[i]);
    }

    vector<double> getWeightDeltaListForPrevious() {
        vector<double> weightDeltaList(getInputSize(), 0.0);
        vector<double> weightDeltaListPart;
        for (int n_i = 0; n_i < neurons.size(); n_i++) {
            weightDeltaListPart = neurons[n_i].getWeightDeltaListPart();
            for (int wDL_i = 0; wDL_i < weightDeltaList.size(); wDL_i++) weightDeltaList[wDL_i] += (double)weightDeltaListPart[wDL_i] / weightDeltaList.size();  // TODO: Here
        }
        return weightDeltaList;
    }

    void calculateDelta(vector<double> deltaWeightList) {
        for (int n_i = 0; n_i < neurons.size(); n_i++) neurons[n_i].setDelta(deltaWeightList[n_i]);
    }

    void update(double eta) {
        for (int i_n = 0; i_n < neurons.size(); i_n++) {
            // cout << i_n << ' ';
            neurons[i_n].update(eta);
        }
    }
};

class Network {
   private:
    vector<Layer> layers;

   public:
    Network(vector<int> neuronCounts) {
        for (int i = 1; i < neuronCounts.size(); i++) {
            layers.push_back(Layer(neuronCounts[i], neuronCounts[i - 1]));
        }
    }

    vector<double> forward(vector<double> x) {
        vector<double> a = x;
        for (int i = 0; i < layers.size(); i++) a = layers[i].forward(a);
        return a;
    }

    void train(vector<vector<double>> x, vector<vector<double>> y, double eta) {
        for (int i_d = 0; i_d < x.size(); i_d++) {
            vector<double> xI = x[i_d];
            vector<double> yI = y[i_d];

            layers.back().calculateDeltaForOutput(yI);
            for (int l_i = layers.size() - 2; l_i > -1; l_i--) {
                layers[l_i].calculateDelta(layers[l_i + 1].getWeightDeltaListForPrevious());
            }

            for (int l_i = 0; l_i < layers.size(); l_i++) {
                // cout << "Update l" << l_i << ' ';
                layers[l_i].update(eta);
                // cout << endl;
            }
        }
    }
};

void printVector(vector<double> v) {
    cout << '[';
    for (int i = 0; i < v.size(); i++) cout << ' ' << v[i] << ',';
    cout << ']' << endl;
}

int test0() {
    Network network = Network({2, 2, 3});

    vector<double> output;

    output = network.forward({1.0, 1.0});
    printVector(output);

    vector<double> xI = {1.0, 1.0};
    vector<double> yI = {2.0, 3.0, 4.0};

    vector<vector<double>> x = {xI};
    vector<vector<double>> y = {yI};

    int turn = 10000000;
    double eta = 0.0000001;

    for (int i = 1; i < turn + 1; i++) {
        network.train(x, y, eta);
        if (i % (turn / 10) == 0) {
            output = network.forward({1.0, 1.0});
            cout << i << " [";
            for (int j = 0; j < output.size(); j++) cout << ' ' << output[j] << ',';
            cout << "] error = ";
            double error = 0.0;
            for (int k = 0; k < yI.size(); k++) error += pow((yI[k] - output[k]), 2);
            cout << error << endl;
        }
    }

    output = network.forward({1.0, 1.0});
    printVector(output);
    cout << "Target" << endl;
    printVector(yI);

    cin;
    return 0;
}

tuple<vector<vector<double>>, vector<vector<double>>> getDataset() {
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

    vector<vector<double>> X;
    vector<vector<double>> y;  // TODO: Make y 2d
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

void test1() {
    vector<vector<double>> x, y_2d;
    vector<double> y_1d;

    tie(x, y_2d) = getDataset();
    for (int i = 0; i < y_2d.size(); i++)
        for (int j = 0; j < y_2d[0].size(); j++) y_1d.push_back(y_2d[i][j]);

    vector<vector<double>> y;
    for (int i = 0; i < y_1d.size(); i++) y.push_back({y_1d[i]});

    Network network = Network({5, 5, 1});

    vector<double> output;

    output = network.forward({1.0, 1.0});
    printVector(output);

    int turn = 100;
    double eta = 0.000001;
    double error = 0.0;

    for (int i = 1; i < turn + 1; i++) {
        network.train(x, y, eta);
        output = network.forward({1.0, 1.0});
        cout << "Turn " << i << " [";
        for (int i = 0; i < output.size(); i++) cout << ' ' << output[i] << ',';
        cout << "] error = ";
        error = 0;
        for (int j = 0; j < y[i].size(); j++) error += pow((y[i][j] - output[j]), 2);
        cout << error << endl;
    }

    cin;
}

void test2() {
    vector<vector<double>> x, y;
    tie(x, y) = getDataset();
    cout << "x size: " << x.size() << " x " << x[0].size() << endl;
    cout << "y size: " << y.size() << endl;

    vector<vector<double>> xTrain, xTest, yTrain, yTest;
    int iSplit = x.size() * 0.8;

    for (int i = 0; i < iSplit; i++) {
        xTrain.push_back(x[i]);
        yTrain.push_back(y[i]);
    }
    cout << "xTrain size: " << xTrain.size() << " x " << xTrain[0].size() << endl;
    cout << "yTrain size: " << yTrain.size() << endl;

    for (int i = iSplit; i < x.size(); i++) {
        xTest.push_back(x[i]);
        yTest.push_back(y[i]);
    }
    cout << "xTest size: " << xTest.size() << " x " << xTest[0].size() << endl;
    cout << "yTest size: " << yTest.size() << endl;

    cout << endl;

    Network network = Network({int(x.front().size()), 2, 1});

    vector<double> yPred;
    double cost = 0;

    for (int i = 0; i < 1000; i++) {
        network.train(xTrain, yTrain, 0.000001);
        for (int j = 0; j < xTest.size(); j++) {
            yPred = network.forward(xTest[j]);
            for (int k = 0; k < yTest[0].size(); k++) {
                cost += pow(yTest[j][k] - yPred[k], 2);
            }
        }
    }
}

void singleLineDummy() {
    cout << randomDouble() << endl;
    vector<vector<double>> x, y;
    x = {{0.1, 0.2, 0.3}};
    y = {{0.5}};
    int epoch = 1e+5;
    double learningRate = 1e-2;

    Network network = Network({3, 64, 32, 16, 1});

    cout << network.forward(x[0])[0] << endl;
    for (int i = 0; i < epoch; i++) {
        network.train(x, y, learningRate);
        cout << network.forward(x[0])[0] << endl;
    }
}

int main() {
    singleLineDummy();

    return 0;
}