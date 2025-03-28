#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>

using namespace std;

const vector<vector<float>> edgeKernel = {
    {1, 0, -1},
    {1, 0, -1},
    {1, 0, -1}
};
vector<vector<float>> padImage(const vector<vector<float>>& image, int padSize) {
    int paddedSize = image.size() + 2 * padSize;
    vector<vector<float>> paddedImage(paddedSize, vector<float>(paddedSize, 0));
    
    for (int i = 0; i < image.size(); ++i)
        for (int j = 0; j < image[0].size(); ++j)
            paddedImage[i + padSize][j + padSize] = image[i][j];
    
    return paddedImage;
}

vector<vector<float>> convolve(const vector<vector<float>>& image, const vector<vector<float>>& kernel) {
    int M = image.size();
    int N = image[0].size();
    int K = kernel.size();
    int padSize = K / 2;

    vector<vector<float>> paddedImage = padImage(image, padSize);
    vector<vector<float>> output(M, vector<float>(N, 0));

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int ki = 0; ki < K; ++ki) {
                for (int kj = 0; kj < K; ++kj) {
                    sum += paddedImage[i + ki][j + kj] * kernel[ki][kj];
                }
            }
            output[i][j] = sum;
        }
    }
    return output;
}

vector<vector<float>> loadImage(const string& filename, int size) {
    vector<vector<float>> image(size, vector<float>(size));
    ifstream file(filename);
    string line;
    int i = 0;
    while (getline(file, line) && i < size) {
        stringstream ss(line);
        for (int j = 0; j < size; ++j) {
            ss >> image[i][j];
        }
        ++i;
    }
    return image;
}

void saveImage(const string& filename, const vector<vector<float>>& image) {
    ofstream file(filename);
    for (const auto& row : image) {
        for (float val : row)
            file << val << " ";
        file << "\n";
    }
}

int main() {
    int imageSize = 512; // Example size (adjustable)
    string inputFile = "image_input.txt";
    string outputFile = "image_output.txt";
    
    vector<vector<float>> image = loadImage(inputFile, imageSize);
    
    auto start = chrono::high_resolution_clock::now();
    vector<vector<float>> output = convolve(image, edgeKernel);
    auto end = chrono::high_resolution_clock::now();
    
    chrono::duration<double> elapsed = end - start;
    cout << "Execution Time: " << elapsed.count() << " seconds\n";
    
    saveImage(outputFile, output);
    
    return 0;
}
