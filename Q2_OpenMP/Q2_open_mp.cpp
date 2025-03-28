#include <iostream>
#include <cmath>
#include <vector>
#include <SDL2/SDL.h>
#include <omp.h>

#define WIDTH  800
#define HEIGHT 600
#define ITERATIONS 12 
#define PI 3.14159265358979323846

using namespace std;

double sin_taylor(double x) {
    double term = x, sum = x;
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        int l_sum=0;
        #pragma omp for schedule(static)

        for (int n = 1; n < ITERATIONS; n++) {
            term *= -x * x / ((2 * n) * (2 * n + 1));
            l_sum += term;
        }
        #pragma omp atomic
        sum += l_sum;
    }
    return sum;
}

double cos_taylor(double x) {
    double term = 1, sum = 1;
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        int l_sum=0;
        #pragma omp for schedule(static)

        for (int n = 1; n < ITERATIONS; n++) {
            term *= -x * x / ((2 * n - 1) * (2 * n));
            l_sum += term;
        }
        #pragma omp atomic
        sum += l_sum;
    }
    
    return sum;
}

vector<pair<int, int>> compute_circle(int j, int k, int r) {
    vector<pair<int, int>> points(360);
    double t_start = omp_get_wtime();

    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (int t = 0; t < 360; t++) {
            double rad = t * (PI / 180.0);
            double x = r * cos_taylor(rad) + j;
            double y = r * sin_taylor(rad) + k;
            points[t] = {static_cast<int>(x), static_cast<int>(y)};
        }
        
        
    }

    double t_end = omp_get_wtime();
    cout << "Parallel Execution Time: " << (t_end - t_start) << " seconds" << endl;
    return points;
}

void draw_circle(vector<pair<int, int>> &points) {
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("OpenMP Circle", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);
    
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    for (auto &point : points) {
        SDL_RenderDrawPoint(renderer, point.first, point.second);
    }

    SDL_RenderPresent(renderer);
    SDL_Delay(5000);
    
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

int main() {
    int j = WIDTH / 2, k = HEIGHT / 2, r = 200;

    vector<pair<int, int>> points = compute_circle(j, k, r);
    draw_circle(points);

    return 0;
}
