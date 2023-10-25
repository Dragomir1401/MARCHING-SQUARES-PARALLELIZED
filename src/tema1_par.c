// Author: APD team, except where source was noted

#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <assert.h>

#define CONTOUR_CONFIG_COUNT 16
#define FILENAME_MAX_SIZE 50
#define STEP 8
#define SIGMA 200
#define RESCALE_X 2048
#define RESCALE_Y 2048

typedef struct
{
    int *thread_id;
    int start_rescale;
    int end_rescale;
    int start_grid_x;
    int end_grid_x;
    int start_grid_y;
    int end_grid_y;
    char *out_file;
    unsigned char ***grid;
    ppm_image ***contour_map;
    ppm_image **image;
    ppm_image **rescaled_image;
    pthread_barrier_t *barrier;
} thread_partition;

#define CLAMP(v, min, max) \
    if (v < min)           \
    {                      \
        v = min;           \
    }                      \
    else if (v > max)      \
    {                      \
        v = max;           \
    }

int min(int a, int b)
{
    if (a < b)
    {
        return a;
    }
    return b;
}

// Creates a map between the binary configuration (e.g. 0110_2) and the corresponding pixels
// that need to be set on the output image. An array is used for this map since the keys are
// binary numbers in 0-15. Contour images are located in the './contours' directory.
ppm_image **init_contour_map()
{
    ppm_image **map = (ppm_image **)malloc(CONTOUR_CONFIG_COUNT * sizeof(ppm_image *));
    if (!map)
    {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++)
    {
        char filename[FILENAME_MAX_SIZE];
        sprintf(filename, "./contours/%d.ppm", i);
        map[i] = read_ppm(filename);
    }

    return map;
}

// Updates a particular section of an image with the corresponding contour pixels.
// Used to create the complete contour image.
void update_image(ppm_image **image, ppm_image *contour, int x, int y)
{
    for (int i = 0; i < contour->x; i++)
    {
        for (int j = 0; j < contour->y; j++)
        {
            int contour_pixel_index = contour->x * i + j;
            int image_pixel_index = (x + i) * (*image)->y + y + j;

            (*image)->data[image_pixel_index].red = contour->data[contour_pixel_index].red;
            (*image)->data[image_pixel_index].green = contour->data[contour_pixel_index].green;
            (*image)->data[image_pixel_index].blue = contour->data[contour_pixel_index].blue;
        }
    }
}

// Corresponds to step 1 of the marching squares algorithm, which focuses on sampling the image.
// Builds a p x q grid of points with values which can be either 0 or 1, depending on how the
// pixel values compare to the `sigma` reference value. The points are taken at equal distances
// in the original image, based on the `step_x` and `step_y` arguments.
void sample_grid(thread_partition *partition)
{
    unsigned char **grid = (*partition->grid);
    ppm_image *image = (*partition->image);

    int startX = partition->start_grid_x;
    int endX = partition->end_grid_x;
    int startY = partition->start_grid_y;
    int endY = partition->end_grid_y;

    int p = image->x / STEP;
    int q = image->y / STEP;

    int p1 = startX / STEP;
    int q1 = startY / STEP;

    int p2 = endX / STEP;
    int q2 = endY / STEP;

    for (int i = p1; i < p2; i++)
    {
        for (int j = 0; j < q; j++)
        {
            ppm_pixel curr_pixel = image->data[i * STEP * image->y + j * STEP];

            unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

            if (curr_color > SIGMA)
            {
                grid[i][j] = 0;
            }
            else
            {
                grid[i][j] = 1;
            }
        }
    }

    // last sample points have no neighbors below / to the right, so we use pixels on the
    // last row / column of the input image for them
    for (int i = p1; i < p2; i++)
    {
        ppm_pixel curr_pixel = image->data[i * STEP * image->y + image->x - 1];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > SIGMA)
        {
            grid[i][q] = 0;
        }
        else
        {
            grid[i][q] = 1;
        }
    }

    for (int j = q1; j < q2; j++)
    {
        ppm_pixel curr_pixel = image->data[(image->x - 1) * image->y + j * STEP];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > SIGMA)
        {
            grid[p][j] = 0;
        }
        else
        {
            grid[p][j] = 1;
        }
    }
}

// Corresponds to step 2 of the marching squares algorithm, which focuses on identifying the
// type of contour which corresponds to each subgrid. It determines the binary value of each
// sample fragment of the original image and replaces the pixels in the original image with
// the pixels of the corresponding contour image accordingly.
void march(thread_partition *partition)
{
    ppm_image **image = partition->image;
    ppm_image **contour_map = (*partition->contour_map);
    unsigned char **grid = (*partition->grid);
    int start = partition->start_grid_x;
    int end = partition->end_grid_x;

    int p1 = start / STEP;
    int p2 = end / STEP;
    int q = (*image)->y / STEP;

    for (int i = p1; i < p2; i++)
    {
        for (int j = 0; j < q; j++)
        {
            unsigned char k = 8 * grid[i][j] + 4 * grid[i][j + 1] + 2 * grid[i + 1][j + 1] + 1 * grid[i + 1][j];
            update_image(image, contour_map[k], i * STEP, j * STEP);
        }
    }
}

// Calls `free` method on the utilized resources.
void free_resources(ppm_image *image, ppm_image **contour_map, unsigned char **grid, int step_x)
{
    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++)
    {
        free(contour_map[i]->data);
        free(contour_map[i]);
    }
    free(contour_map);

    for (int i = 0; i <= image->x / step_x; i++)
    {
        free(grid[i]);
    }
    free(grid);
}

void rescale_image(thread_partition *partition)
{
    ppm_image *new_image = (*partition->rescaled_image);
    ppm_image **image = partition->image;
    int start = partition->start_rescale;
    int end = partition->end_rescale;

    // We only rescale downwards
    if ((*image)->x <= RESCALE_X && (*image)->y <= RESCALE_Y)
    {
        return;
    }

    uint8_t sample[3];

    // Use bicubic interpolation for scaling
    for (int i = start; i < end; i++)
    {
        for (int j = 0; j < new_image->y; j++)
        {
            float u = (float)i / (float)(new_image->x - 1);
            float v = (float)j / (float)(new_image->y - 1);
            sample_bicubic((*image), u, v, sample);

            new_image->data[i * new_image->y + j].red = sample[0];
            new_image->data[i * new_image->y + j].green = sample[1];
            new_image->data[i * new_image->y + j].blue = sample[2];
        }
    }
}

void *thread_routine(void *arg)
{
    // Cast the argument to thread_partition_array
    thread_partition *partition = (thread_partition *)arg;

    printf("Thread %d: start_rescale = %d, end_rescale = %d, start_grid_x = %d, end_grid_x = %d, start_grid_y = %d, end_grid_y = %d\n", *partition->thread_id, partition->start_rescale, partition->end_rescale, partition->start_grid_x, partition->end_grid_x, partition->start_grid_y, partition->end_grid_y);

    // 1. Rescale the image
    rescale_image(partition);

    pthread_barrier_wait(partition->barrier);

    // 2. Sample the grid
    sample_grid(partition);

    pthread_barrier_wait(partition->barrier);

    // 3. March the squares
    march(partition);

    pthread_barrier_wait(partition->barrier);

    // 4. Write output
    write_ppm((*partition->rescaled_image), partition->out_file);

    return NULL;
}

pthread_barrier_t *alloc_barrier(int P)
{
    // Allocate memory for the barrier
    pthread_barrier_t *barrier;
    barrier = (pthread_barrier_t *)malloc(sizeof(pthread_barrier_t));
    pthread_barrier_init(barrier, NULL, P);

    return barrier;
}

ppm_image *alloc_new_image()
{
    // Alloc memory for rescaled image
    ppm_image *new_image = (ppm_image *)malloc(sizeof(ppm_image));
    if (!new_image)
    {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    new_image->x = RESCALE_X;
    new_image->y = RESCALE_Y;

    new_image->data = (ppm_pixel *)malloc(new_image->x * new_image->y * sizeof(ppm_pixel));
    if (!new_image)
    {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    return new_image;
}

unsigned char **alloc_grid(ppm_image *new_image, ppm_image *image)
{
    int p = new_image->x / STEP;
    int q = new_image->y / STEP;

    // We only rescale downwards
    if (image->x <= RESCALE_X && image->y <= RESCALE_Y)
    {
        p = image->x / STEP;
        q = image->y / STEP;
    }

    // Allocate memory for the grid
    unsigned char **grid = (unsigned char **)malloc((p + 1) * sizeof(unsigned char *));
    if (!grid)
    {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i <= p; i++)
    {
        grid[i] = (unsigned char *)malloc((q + 1) * sizeof(unsigned char));
        if (!grid[i])
        {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }
    }

    return grid;
}

thread_partition *populate_partition(
    int i,
    int P,
    ppm_image **image,
    ppm_image **new_image,
    unsigned char ***grid,
    pthread_barrier_t *barrier,
    ppm_image ***contour_map,
    char **argv,
    int *thread_id)
{
    thread_partition *result;
    result = (thread_partition *)malloc(sizeof(thread_partition));

    result->grid = grid;
    result->image = image;
    result->barrier = barrier;
    result->contour_map = contour_map;
    result->rescaled_image = new_image;
    result->out_file = argv[2];
    result->thread_id = &thread_id[i];

    result->start_rescale = i * (double)RESCALE_X / P;
    result->end_rescale = min((i + 1) * (double)RESCALE_X / P, RESCALE_X);

    result->start_grid_x = i * (double)((*image)->x) / P;
    result->end_grid_x = min((i + 1) * (*image)->x / P, (*image)->x);

    result->start_grid_y = i * (double)(*image)->y / P;
    result->end_grid_y = min((i + 1) * (double)(*image)->y / P, (*image)->y);

    return result;
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        fprintf(stderr, "Usage: ./tema1 <in_file> <out_file> <P>\n");
        return 1;
    }

    // Create P threads using pthread API from POSIX
    int P = atoi(argv[3]);
    pthread_t tid[P];
    int thread_id[P];

    // Allocate memory for the barrier
    pthread_barrier_t *barrier = alloc_barrier(P);

    // Allocate memory for the thread partitions
    thread_partition **partition = (thread_partition **)malloc(P * sizeof(thread_partition *));

    // Read input image
    ppm_image *image = read_ppm(argv[1]);

    // Create contour map
    ppm_image **contour_map = init_contour_map();

    // Alloc memory for rescaled image
    ppm_image *new_image;
    if (image->x <= RESCALE_X && image->y <= RESCALE_Y)
    {
        new_image = image;
    }
    else
    {
        new_image = alloc_new_image();
    }

    // Allocate memory for the grid
    unsigned char **grid = alloc_grid(new_image, image);

    // Create the threads
    for (int i = 0; i < P; i++)
    {
        thread_id[i] = i;
        partition[i] = populate_partition(i, P, &image, &new_image, &grid, barrier, &contour_map, argv, &thread_id[i]);

        int rc = pthread_create(&(tid[i]), NULL, thread_routine, partition[i]);
        if (rc)
        {
            fprintf(stderr, "Error: unable to create thread, %d\n", rc);
            exit(-1);
        }
    }

    // Wait for the threads to finish
    for (int i = 0; i < P; i++)
    {
        int rc = pthread_join(tid[i], NULL);
        if (rc)
        {
            fprintf(stderr, "Error: unable to join, %d\n", rc);
            exit(-1);
        }
    }

    for (int i = 0; i < P; i++)
    {
        free(partition[i]);
    }

    // Free rescalled image, contour map and grid
    free_resources(new_image, contour_map, grid, STEP);

    // Free the image
    free(image->data);
    free(image);

    // Free the thread partition array
    free(partition);

    // Free barrier
    pthread_barrier_destroy(barrier);
    free(barrier);

    return 0;
}
