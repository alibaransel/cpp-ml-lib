#include <iostream>
#include <vector>

using namespace std;

class Neuron {
   private:
    vector<double> w;
    double b;

    double activation(double z) {
        return z;
    }

   public:
    Neuron(int inputSize) {
        for (int i = 0; i < inputSize; i++) w.push_back(1.0);
        b = 1.0;
    }

    double forward(vector<double> x) {
        double z = b;
        for (int i = 0; i < w.size(); i++) z += w[i] * x[i];
        return activation(z);
    }
};

class Layer {
   private:
    vector<Neuron> neurons;

   public:
    Layer(int neuronCount, int inputSize) {
        for (int i = 0; i < neuronCount; i++) neurons.push_back(Neuron(inputSize));
    }

    vector<double> forward(vector<double> x) {
        vector<double> a;
        for (int i = 0; i < neurons.size(); i++) a.push_back(neurons[i].forward(x));
        return a;
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
};

int main() {
    Network network = Network({2, 2, 3});
    vector<double> output = network.forward({1.0, 1.0});
    cout << '[';
    for (int i = 0; i < output.size(); i++) cout << ' ' << output[i] << ',';
    cout << ']' << endl;
    cin;
    return 0;
}