#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

class Neuron {
   private:
    vector<double> w;
    double b;

    vector<double> aIn;

    double z;
    double a;
    double delta_z = 0.0;

    double activation(double z) {
        return z;
    }

    double d_activation() {
        return 1.0;
    }

   public:
    Neuron(int inputSize) {
        for (int i = 0; i < inputSize; i++) {
            w.push_back(0.5);
        }
        b = 0.5;
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
            for (int wDL_i = 0; wDL_i < weightDeltaList.size(); wDL_i++) weightDeltaList[wDL_i] += weightDeltaListPart[wDL_i];
        }
        return weightDeltaList;
    }

    void calculateDelta(vector<double> deltaWeightList) {
        for (int n_i = 0; n_i < neurons.size(); n_i++) neurons[n_i].setDelta(deltaWeightList[n_i]);
    }

    void update(double eta) {
        for (int i_n = 0; i_n < neurons.size(); i_n++) neurons[i_n].update(eta);
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

            for (int l_i = 0; l_i < layers.size(); l_i++) layers[l_i].update(eta);
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

    int turn = 10000000;
    double eta = 0.0000001;

    for (int i = 1; i < turn + 1; i++) {
        network.train({xI}, {yI}, eta);
        if (i % (turn / 10) == 0) {
            output = network.forward({1.0, 1.0});
            cout << '[';
            for (int i = 0; i < output.size(); i++) cout << ' ' << output[i] << ',';
            cout << "] error = ";
            double error = 0.0;
            for (int j = 0; j < yI.size(); j++) error += pow((yI[j] - output[j]), 2);
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

int main() {
    test0();
    cin;
    return 0;
}