#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace std;

// Linear Regression Model
/*
template <typename T>
using Matrix = vector<vector<T>>;

void showError(string text) {
    cout << "Encounter an error!" << endl;
}



Matrix<string> readCSV() {}

tuple<vector<typename T>, Matrix<typename A> convertDataset() {
    return {vector(2, 0), Matrix<>

}



double sigmoid() {}

double dotProduct(vector<double> v1, vector<double> v2) {
    if(v1.size() != v2.size()) showError();
}
*/

// Scalar Product
vector<double> operator*(double s, vector<double> v)
{
    for (int i = 0; i < v.size(); i++)
        v[i] *= s;
    return v;
}

// Scalar Division
vector<double> operator/(vector<double> v, double s)
{
    for (int i = 0; i < v.size(); i++)
        v[i] /= s;
    return v;
}
/*
vector<double> & operator/=(vector<double> &v, double s) {
    for (int i = 0; i < v.size(); i++) v[i] /= s;
    return v;
}
*/
// Dot Subtraction
double operator*(vector<double> v1, vector<double> v2)
{
    double res = 0;
    for (int i = 0; i < v1.size(); i++)
        res += v1[i] * v2[i];
    return res;
}

// Vector Subtraction
vector<double> operator-(vector<double> v1, vector<double> v2)
{
    vector<double> v(v1.size());
    for (int i = 0; i < v1.size(); i++)
        v[i] = v1[i] - v2[i];
    return v;
}

double func(vector<double> w, double b, vector<double> x)
{
    return w * x + b;
}

double computeCost(vector<vector<double>> X, vector<double> y, vector<double> w, double b)
{
    int m = X.size();
    int n = X[0].size();
    double cost = 0;
    double diff = 0;
    for (int i = 0; i < m; i++)
    {
        diff = func(w, b, X[i]) - y[i];
        diff *= diff;
        cost += diff;
        diff = 0;
    }
    return cost;
}

tuple<vector<double>, double> computeGradient(vector<vector<double>> X, vector<double> y, vector<double> w, double b)
{
    int m = X.size();
    int n = X[0].size();

    vector<double> djdw(n, 0);
    double djdb = 0;

    double loss = 0;
    for (int i = 0; i < m; i++)
    {
        loss = func(w, b, X[i]) - y[i];
        for (int j = 0; j < n; j++)
        {
            djdw[j] += loss * X[i][j]; // error
        }
        djdb += loss;
    }
    djdw = djdw / m; // change later
    djdb /= m;
    return {djdw, djdb};
}

tuple<vector<double>, double> runGradientDescent(vector<vector<double>> X, vector<double> y, vector<double> w, double b, double alpha, int epoch)
{
    vector<double> djdw;
    double djdb;
    double cost;
    cout << fixed << setfill('0');
    for (int i = 0; i < epoch; i++)
    {
        tie(djdw, djdb) = computeGradient(X, y, w, b);
        w = w - alpha * djdw;
        b = b - alpha * djdb;
        cost = computeCost(X, y, w, b);
        cout << "Epoc " << setw(6) << i << " w = [";
        for (int j = 0; j < w.size(); j++)
            cout << setw(8) << w[j] << ", ";
        cout << "] b = " << b << " Cost = " << setw(8) << cost << endl;
    }
    return {w, b};
}

tuple<vector<vector<double>>, vector<double>> getDataset()
{
    ifstream file("Student_Performance.csv");
    if (!file.is_open())
        exit(0); // improve later
    string line;
    getline(file, line);
    istringstream lineSS(line);
    vector<string> keys;
    string key;
    while (getline(lineSS, key, ','))
        keys.push_back(key);

    vector<vector<double>> X;
    vector<double> y;
    vector<double> lineData;
    string stringData;
    int n = 10;
    while (n > 0 && getline(file, line))
    {
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
        y.push_back(stod(stringData) / 100);
        X.push_back(lineData);
        lineData.clear();
    }
    file.close();
    return {X, y};
}

int main()
{
    // prepare dataset
    // read dataset
    // convert dataset
    // scale dataset
    // shuffle dataset

    // split dataset

    // train model

    // test model

    vector<vector<double>> X;
    vector<double> y;
    tie(X, y) = getDataset();
    int m = X.size();
    int n = X[0].size();
    vector<double> w(n, 1);
    double b;
    /*
    w[0] = 0.294291;
    w[1] = 1.031112;
    w[2] = 0.008468;
    w[3] = 0.114809;
    w[4] = 0.025228;
    b = -0.402368;
    */
    const double alpha = 0.001;
    const int epoch = 10000;
    tie(w, b) = runGradientDescent(X, y, w, b, alpha, epoch);
    cout << "w = ";
    for (int i = 0; i < w.size(); i++)
        cout << w[i] << ", ";
    cout << endl;
    cout << "b = " << b << endl;
    cout << endl;
    cout << "precict <> actual" << endl;
    for (int i = 0; i < 10; i++)
        cout << w * X[i] + b << " <> " << y[i] << endl;
    scanf("%d");
}

// Epoc 000999 w = [0.588280, 0.480379, 0.462658, 0.485103, 0.590425, ] b = -0.764246 Cost = 1162.385565
// Epoc 009999 w = [0.488315, 0.674316, 0.079329, 0.452688, 0.326093, ] b = -0.655315 Cost = 207.552575