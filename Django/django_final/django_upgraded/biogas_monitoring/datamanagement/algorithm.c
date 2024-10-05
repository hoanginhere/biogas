#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <float.h>
#define POP_SIZE 10
#define MAX_T 20
#define ALGORITHM_ITERIAL 200
#define NUM_HOURS  48
#define NUM_LOADS  9
#define ROWS 9  // Số hàng trong mảng
#define COLS 48 // Số cột trong mảng
double exhaustFan[48] = {26.2, 26.2, 26.2, 26.2, 26.2, 26.2,26.2, 26.2, 26.2, 26.2, 26.2, 26.2, 52.4, 52.4, 52.4, 52.4, 52.4, 52.4,
                                 52.4, 52.4, 52.4, 52.4, 52.4, 52.4, 52.4, 52.4, 52.4, 52.4,
                                 52.4, 52.4, 52.4, 52.4,26.2, 26.2, 26.2, 26.2, 26.2, 26.2,26.2, 26.2, 26.2, 26.2, 26.2, 26.2};
double wastewaterTreatment[48] = {5};
double coolingPump[48] = {8.8, 8.8, 8.8, 8.8, 8.8, 8.8,8.8, 8.8, 8.8, 8.8, 8.8, 8.8, 17.6, 17.6, 17.6, 17.6, 17.6, 17.6,
                                   17.6, 17.6, 17.6, 17.6,7.6, 17.6, 17.6, 17.6, 17.6, 17.6,
                                   17.6, 17.6, 17.6, 17.6, 8.8, 8.8, 8.8, 8.8, 8.8, 8.8,8.8, 8.8, 8.8, 8.8, 8.8, 8.8};
double lighting[48] = {4.4};
double aeration[48] = {10};
double compensationPump[48] = {2};
double bioGen1[48] = {60};
double bioGen2[48] = {80};
double bioGen3[48] = {100};
double LCOE1 = 1000;
double LCOE2 = 1200;
double LCOE3 = 1600;
double EP[48] = {1044, 1044, 1044, 1044, 1044, 1044, 1044, 1044,1649, 1649, 1649, 1649, 1649, 1649,1649, 1649, 1649, 1649, 1649, 1649, 2973, 2973,2973, 2973,
                        1649, 1649, 1649,1649, 1649, 1649,1649, 1649, 1649,1649, 2973, 2973, 2973,2973, 2973, 2973, 1649, 1649,1649, 1649, 1044, 1044,1044, 1044};
double position[9][48];
double exhaustFanLoad[48], wastewaterTreatmentLoad[48];
double coolingPumpLoad[48], lightingLoad[48];
double aerationLoad[48], compensationPumpLoad[48];
double loadDemand[48];
double bioPowerCapacity1[48], bioPowerCapacity2[48], bioPowerCapacity3[48];
double totalBioPowerCapacity[48], EWithBio[48];
double LWithBio = 0;
double sum(double* array, int size) {
    double total = 0.0;
    for (int i = 0; i < size; i++) {
        total += array[i];
    }
    return total;
}
void printPosition(double position[ROWS][COLS]) {
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%lf ", position[i][j]); // In giá trị tại từng ô
        }
        printf("\n"); 
    }
}
struct schedule {
    int index;                // Chỉ số thiết bị
    char schedule_option[30];  // Tùy chọn lịch (yes/no)
    int exact_hours;          // Số giờ cụ thể
    char schedule[30]; // Lịch trình hoạt động (full_day/night)
    char run_time[30]; //chế độ chạy trong ngày
    char power_mode[30];      // Chế độ hoạt động (normal)
};
// Function to compute cost without biogas
double costWithoutBio(double position[9][48]) {
    double exhaustFanLoad[48], wastewaterTreatmentLoad[48];
    double coolingPumpLoad[48], lightingLoad[48];
    double aerationLoad[48], compensationPumpLoad[48];
    double loadDemand[48];
    double LWithoutBio = 0;

    // Compute Load Demand
    for (int i = 0; i < 48; i++) {
        exhaustFanLoad[i] = exhaustFan[i] * position[0][i];
        wastewaterTreatmentLoad[i] = wastewaterTreatment[i] * position[1][i];
        coolingPumpLoad[i] = coolingPump[i] * position[2][i];
        lightingLoad[i] = lighting[i] * position[3][i];
        aerationLoad[i] = aeration[i] * position[4][i];
        compensationPumpLoad[i] = compensationPump[i] * position[5][i];
        loadDemand[i] = exhaustFanLoad[i] + wastewaterTreatmentLoad[i] +
                         coolingPumpLoad[i] + lightingLoad[i] +
                         aerationLoad[i] + compensationPumpLoad[i];
    }

    // Compute Cost
    for (int i = 0; i < 48; i++) {
        LWithoutBio += loadDemand[i] * EP[i];
    }
    return LWithoutBio;
}

// Function to compute objective function
double ObjF(double position[9][48]) {
    double exhaustFanLoad[48], wastewaterTreatmentLoad[48];
    double coolingPumpLoad[48], lightingLoad[48];
    double aerationLoad[48], compensationPumpLoad[48];
    double loadDemand[48];
    double bioPowerCapacity1[48], bioPowerCapacity2[48], bioPowerCapacity3[48];
    double totalBioPowerCapacity[48], EWithBio[48];
    double LWithBio = 0;

    // Compute Load Demand and Biogas Generator Capacity
    for (int i = 0; i < 48; i++) {
        exhaustFanLoad[i] = exhaustFan[i] * position[0][i];
        wastewaterTreatmentLoad[i] = wastewaterTreatment[i] * position[1][i];
        coolingPumpLoad[i] = coolingPump[i] * position[2][i];
        lightingLoad[i] = lighting[i] * position[3][i];
        aerationLoad[i] = aeration[i] * position[4][i];
        compensationPumpLoad[i] = compensationPump[i] * position[5][i];
        loadDemand[i] = exhaustFanLoad[i] + wastewaterTreatmentLoad[i] +
                         coolingPumpLoad[i] + lightingLoad[i] +
                         aerationLoad[i] + compensationPumpLoad[i];

        bioPowerCapacity1[i] = position[6][i] * bioGen1[i];
        bioPowerCapacity2[i] = position[7][i] * bioGen2[i];
        bioPowerCapacity3[i] = position[8][i] * bioGen3[i];
        totalBioPowerCapacity[i] = bioPowerCapacity1[i] + bioPowerCapacity2[i] + bioPowerCapacity3[i];

        // Compute Total Energy using Biogas
        EWithBio[i] = loadDemand[i] - totalBioPowerCapacity[i];
        if (EWithBio[i] < 0) {
            EWithBio[i] = 0;
        }
    }

    // Compute Cost
    for (int i = 0; i < 48; i++) {
        LWithBio += EWithBio[i] * EP[i];
    }
    for (int i = 0; i < 48; i++) {
        LWithBio += bioPowerCapacity1[i] * LCOE1;
        LWithBio += bioPowerCapacity2[i] * LCOE2;
        LWithBio += bioPowerCapacity3[i] * LCOE3;
    }

    return LWithBio;
}
// Function to compute array value
double arrayValue(double position[9][NUM_HOURS], double result[48]) {
    double exhaustFanLoad[48], wastewaterTreatmentLoad[48];
    double coolingPumpLoad[48], lightingLoad[48];
    double aerationLoad[48], compensationPumpLoad[48];
    double loadDemand[48];
    double bioPowerCapacity1[48], bioPowerCapacity2[48], bioPowerCapacity3[48];
    double totalBioPowerCapacity[48], EWithBio[48];

    // Compute Load Demand and Biogas Generator Capacity
    for (int i = 0; i < NUM_HOURS; i++) {
        exhaustFanLoad[i] = exhaustFan[i] * position[0][i];
        wastewaterTreatmentLoad[i] = wastewaterTreatment[i] * position[1][i];
        coolingPumpLoad[i] = coolingPump[i] * position[2][i];
        lightingLoad[i] = lighting[i] * position[3][i];
        aerationLoad[i] = aeration[i] * position[4][i];
        compensationPumpLoad[i] = compensationPump[i] * position[5][i];
        loadDemand[i] = exhaustFanLoad[i] + wastewaterTreatmentLoad[i] +
                         coolingPumpLoad[i] + lightingLoad[i] +
                         aerationLoad[i] + compensationPumpLoad[i];

        bioPowerCapacity1[i] = position[6][i] * bioGen1[i];
        bioPowerCapacity2[i] = position[7][i] * bioGen2[i];
        bioPowerCapacity3[i] = position[8][i] * bioGen3[i];
        totalBioPowerCapacity[i] = bioPowerCapacity1[i] + bioPowerCapacity2[i] + bioPowerCapacity3[i];

        // Compute Total Energy using Biogas
        EWithBio[i] = loadDemand[i] - totalBioPowerCapacity[i];
        if (EWithBio[i] < 0) {
            EWithBio[i] = 0;
        }

        // Compute Cost
        result[i] = EWithBio[i] * EP[i] + bioPowerCapacity1[i] * LCOE1 +
                    bioPowerCapacity2[i] * LCOE2 + bioPowerCapacity3[i] * LCOE3;
    }
    return result[NUM_HOURS];
}

// Define Particle structure
typedef struct {
    double position[9][NUM_HOURS];
    double totalValue;
    double arrayValue[48];
} Particle;

void initParticle(Particle* p, double position[9][48]) {
    // Gán giá trị cho position
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 48; j++) {
            p->position[i][j] = position[i][j];
        }
    }

    // Tính toán totalValue 
    p->totalValue = ObjF(p->position); 

    // Tính toán giá trị arrayValue
    arrayValue(p->position, p->arrayValue);
}



// Function to initialize random position
void randomInitPosition(double position[9][48]) {
    // Khởi tạo bộ sinh số ngẫu nhiên
    srand(time(NULL));

    // Điền mảng với giá trị ngẫu nhiên 0 hoặc 1
    for (int i = 0; i < 18; i++) {
        for (int j = 0; j < 48; j++) {
            position[i][j] = rand() % 2; // Sinh số ngẫu nhiên 0 hoặc 1
        }
    }
}
void checkConstraint(double position[9][48], 
                     struct schedule schedules[6], 
                     int max_runtime, 
                     double max_bio_power, 
                     int max_bio_runtime, 
                     double biogas_power_limits[3]) {
    int max_iterations = 1000;
    // Duyệt qua từng thiết bị để áp dụng lịch trình
    for (int device = 0; device < 6; device++) {
        struct schedule schedule = schedules[device];

        // Khởi tạo trạng thái của thiết bị (24 giờ)
        memset(position[schedule.index], 0, sizeof(double) * 48);

        // Nếu không có yêu cầu lập lịch
        if (strcmp(schedule.schedule_option, "no") == 0) {
            if (strcmp(schedule.run_time, "day") == 0) {
                for (int hour = 12; hour < 36; hour++) {
                    position[schedule.index][hour] = 1;
                }
            } else if (strcmp(schedule.run_time, "night") == 0) {
                for (int hour = 0; hour < 12; hour++) {
                    position[schedule.index][hour] = 1;
                }
                for (int hour = 36; hour < NUM_HOURS; hour++) {
                    position[schedule.index][hour] = 1;
                }
            } else if (strcmp(schedule.run_time, "full_day") == 0) {
                for (int hour = 0; hour < NUM_HOURS; hour++) {
                    position[schedule.index][hour] = 1;
                }
            }
        } else if (strcmp(schedule.schedule_option, "yes") == 0) {
            int total_hours = schedule.exact_hours;
            int start, end;
            if (strcmp(schedule.schedule, "daytime") == 0) {
                start = 12;
                end = 36;
            } else if (strcmp(schedule.schedule, "night") == 0) {
                start = 0;
                end = 12;
                // Đêm có thể là từ 0h-6h và 18h-24h
            } else {
                start = 0;
                end = 48;
            }

            // Chọn ngẫu nhiên số giờ để bật thiết bị
            int hours_to_schedule[total_hours];
            if (strcmp(schedule.schedule, "night") == 0) {
                // Chọn giờ ngẫu nhiên trong khoảng từ 0h-6h và 18h-24h (đêm)
                for (int i = 0; i < total_hours; i++) {
                    hours_to_schedule[i] = rand() % (end - start) + start; // Điều chỉnh cho khoảng 0h-6h và 18h-24h
                }
            } else if (strcmp(schedule.schedule, "daytime") == 0) {
                // Chọn giờ ngẫu nhiên trong khoảng từ 6h-18h (ban ngày)
                for (int i = 0; i < total_hours; i++) {
                    hours_to_schedule[i] = rand() % (end - start) + start; 
                }
            } else {
                // Chọn giờ ngẫu nhiên trong khoảng từ 0h-24h (cả ngày)
                for (int i = 0; i < total_hours; i++) {
                    hours_to_schedule[i] = rand() % 48; 
                }
            }
            // Đánh dấu giờ được lên lịch
            for (int i = 0; i < total_hours; i++) {
                position[schedule.index][hours_to_schedule[i]] = 1;
            }

            int current_runtime = 0;
            for (int hour = 0; hour < 48; hour++) {
                current_runtime += position[schedule.index][hour];
            }
            if (current_runtime > total_hours) {
                // Xóa giờ ngẫu nhiên
                for (int i = 0; i < current_runtime - total_hours; i++) {
                    int hour_to_remove = rand() % 48;
                    position[schedule.index][hour_to_remove] = 0;
                }
            } else if (current_runtime < total_hours) {
                for (int i = 0; i < total_hours - current_runtime; i++) {
                    int hour_to_add = rand() % 48;
                    position[schedule.index][hour_to_add] = 1;
                }
            }
        }
    }

    // Tính toán tổng tải yêu cầu
    double loadDemand[48] = {0};
    for (int hour = 0; hour < 48; hour++) {
        loadDemand[hour] = exhaustFanLoad[hour] + wastewaterTreatmentLoad[hour] +
                           coolingPumpLoad[hour] + lightingLoad[hour] +
                           aerationLoad[hour] + compensationPumpLoad[hour];
    }

    // Kiểm tra và điều chỉnh các máy phát điện Biogas
    for (int hour = 0; hour < NUM_HOURS; hour++) {
        double total_biogas_power = position[6][hour] * bioGen1[hour] + 
                                    position[7][hour] * bioGen2[hour] + 
                                    position[8][hour] * bioGen3[hour];

        if (total_biogas_power < loadDemand[hour]) {
            position[6][hour] = 0;
            position[7][hour] = 0;
            position[8][hour] = 0;
        } else {
            if (bioGen1[hour] >= loadDemand[hour]) {
                position[6][hour] = 1;
                position[7][hour] = 0;
                position[8][hour] = 0;
            } else if (bioGen2[hour] >= loadDemand[hour]) {
                position[6][hour] = 0;
                position[7][hour] = 1;
                position[8][hour] = 0;}
               else if (bioGen3[hour] >= loadDemand[hour]) {
                position[6][hour] = 0;
                position[7][hour] = 0;
                position[8][hour] = 1;
            } else {
                if( bioGen1[hour]+ bioGen2[hour] >= loadDemand[hour])
                {
                    position[6][hour] = 1;
                    position[7][hour] = 1;
                    position[8][hour] = 0;
                }
                else if(bioGen1[hour]+ bioGen3[hour] >= loadDemand[hour])
                {
                    position[6][hour] = 1;
                    position[7][hour] = 0;
                    position[8][hour] = 1;
                }
                else if(bioGen2[hour]+ bioGen3[hour] >= loadDemand[hour])
                {
                    position[6][hour] = 0;
                    position[7][hour] = 1;
                    position[8][hour] = 1;
                }
                else{
                  position[6][hour] = position[7][hour] = position[8][hour] = 1;
                }// Mặc định bật cả ba
            }
        }
    }

    // Kiểm tra giới hạn công suất tổng của biogas
    while (1) {
    // Tính toán công suất biogas của các máy phát
    double bioPowerCapacity1[NUM_HOURS], bioPowerCapacity2[NUM_HOURS], bioPowerCapacity3[NUM_HOURS];
    double totalBioPowerCapacity[NUM_HOURS];

    for (int hour = 0; hour < NUM_HOURS; hour++) {
        bioPowerCapacity1[hour] = position[6][hour] * bioGen1[hour];
        bioPowerCapacity2[hour] = position[7][hour] * bioGen2[hour];
        bioPowerCapacity3[hour] = position[8][hour] * bioGen3[hour];
        totalBioPowerCapacity[hour] = bioPowerCapacity1[hour] + bioPowerCapacity2[hour] + bioPowerCapacity3[hour];
    }

    // Kiểm tra tổng công suất của các máy phát
    double totalBioPowerSum = 0;
    for (int hour = 0; hour < NUM_HOURS; hour++) {
        totalBioPowerSum += totalBioPowerCapacity[hour];
    }

    if (totalBioPowerSum > max_bio_power) {
        // Tìm các chỉ số giờ mà máy phát đang hoạt động (position >= 1)
        int indices[NUM_HOURS];
        int count = 0;
        for (int i = 0; i < NUM_HOURS; i++) {
            if (position[6][i] >= 1 || position[7][i] >= 1 || position[8][i] >= 1) {
                indices[count++] = i;
            }
        }

        if (count == 0) {
            break;
        }

        // Chọn ngẫu nhiên một giờ để tắt máy phát
        int randomIndex = indices[rand() % count];
        position[6][randomIndex] = 0;
        position[7][randomIndex] = 0;
        position[8][randomIndex] = 0;
    } else {
        break;
    }
}

// Kiểm tra giới hạn thời gian hoạt động của biogas
while (1) {
    double totalOperatingHour[NUM_HOURS] = {0};

    for (int hour = 0; hour < NUM_HOURS; hour++) {
        totalOperatingHour[hour] = position[6][hour] + position[7][hour] + position[8][hour];
    }

    double totalOperatingSum = 0;
    for (int hour = 0; hour < NUM_HOURS; hour++) {
        totalOperatingSum += totalOperatingHour[hour];
    }

    if (totalOperatingSum > max_bio_runtime) {
        int indices[NUM_HOURS];
        int count = 0;
        for (int i = 0; i < NUM_HOURS; i++) {
            if (totalOperatingHour[i] >= 1) {
                indices[count++] = i;
            }
        }

        if (count == 0) {
            break;
        }

        // Chọn ngẫu nhiên một giờ để tắt máy phát
        int randomIndex = indices[rand() % count];
        position[6][randomIndex] = 0;
        position[7][randomIndex] = 0;
        position[8][randomIndex] = 0;
    } else {
        break;
    }
}

                     }
void Algorithm(int Pop_Size, int MaxT, struct schedule schedules[6], float max_runtime, float max_bio_power, float max_bio_runtime, double biogas_power_limits[3], Particle *allPositionBest, float *allBestValue) {
    // Khởi tạo các biến
    Particle pbest;
    float bestValue;
    Particle particles[Pop_Size];
    double position[9][48];

    // Generate Random Initial Position
    for (int i = 0; i < Pop_Size; i++) {
        randomInitPosition(position);
        Particle initPos; // Khai báo một biến kiểu Particle
        initParticle(&initPos, position); // Sử dụng hàm initParticle
        checkConstraint(initPos.position, schedules, max_runtime, max_bio_power, max_bio_runtime, biogas_power_limits); // Truyền mảng position
        particles[i] = initPos; // Gán initPos vào mảng particles
    }

    // Check best Position
    for (int i = 0; i < MaxT; i++) {
        float minValue = FLT_MAX;
        int minIndex = -1;

        for (int j = 0; j < Pop_Size; j++) {
            if (particles[j].totalValue < minValue) {
                minValue = particles[j].totalValue;
                minIndex = j;
            }
        }

        pbest = particles[minIndex];
        bestValue = pbest.totalValue;

        // Cập nhật vị trí tốt nhất 
        if (bestValue < *allBestValue) {
            *allBestValue = bestValue;
            *allPositionBest = pbest; // Cập nhật allPositionBest
        }

        //printf("Best position in %d th Iteration: with value %f\n", i, bestValue);
        //printf("Cost without biogas: %f\n", costWithoutBio(pbest.position));

        // Update particle positions
        for (int j = 0; j < Pop_Size; j++) {
            if (j != minIndex) {
                Particle currentPosition = particles[j];
                for (int k = 0; k < 48; k++) {
                    if (currentPosition.arrayValue[k] > pbest.arrayValue[k]) {
                        for (int l = 0; l < 9; l++) {
                            currentPosition.position[l][k] = pbest.position[l][k];
                        }
                    }
                }
                checkConstraint(currentPosition.position, schedules, max_runtime, max_bio_power, max_bio_runtime, biogas_power_limits);
                initParticle(&particles[j], currentPosition.position); // Cập nhật lại particle
            }
        }
    }
}

int main() {
    clock_t start_time = clock();
    float lastBestValue = FLT_MAX;
    Particle lastPositionBest;

    float max_runtime = 12;  
    float max_bio_power = 900;  
    float max_bio_runtime = 15;  
    double biogas_power_limits[3] = {100, 200, 300}; 

    struct schedule schedules[6] = {
        { 0, "no", 0,"none","full_day" ,"normal"},
        { 1, "yes", 24, "full_day","none", "normal"},
        { 2, "yes", 12,"night", "none" ,"normal"},
        { 3, "yes", 12, "night","none","normal"},
        { 4, "no", 0,"full_day", "full_day","normal"},
        { 5, "no",0, "full_day", "full_day","normal"}
    };

    for (int m = 0; m < ALGORITHM_ITERIAL; m++) {
        Particle allPositionBest; // Để lưu trữ vị trí tốt nhất
        float allBestValue = FLT_MAX; // Giá trị tốt nhất
        
        Algorithm(POP_SIZE, MAX_T, schedules, max_runtime, max_bio_power, max_bio_runtime, biogas_power_limits, &allPositionBest, &allBestValue);

        // Kiểm tra giá trị tốt nhất
        if (allBestValue < lastBestValue) {
            lastBestValue = allBestValue;
            lastPositionBest = allPositionBest;
        }
    }
   FILE *fp = fopen("best_position.csv", "w");
    if (fp == NULL) {
        printf("Không thể mở file để ghi!\n");
        return 1;
    }

    // Ghi tiêu đề vào file CSV
    fprintf(fp, "Index\\Hour,");
    for (int hour = 0; hour < 48; hour++) {
        fprintf(fp, "%d,", hour); // In giờ
    }
    fprintf(fp, "\n");

    // Ghi dữ liệu bảng vị trí tốt nhất vào file CSV
    for (int index = 0; index < 9; index++) {
        fprintf(fp, "%d,", index); // In chỉ số hàng
        for (int hour = 0; hour < 48; hour++) {
            fprintf(fp, "%.2f,", lastPositionBest.position[index][hour]); // In giá trị vị trí
        }
        fprintf(fp, "\n");
    }

    // Đóng file
    fclose(fp);

    printf("Best position: ... with value %f\n", lastBestValue); 
    printf("Cost without biogas: %f\n", costWithoutBio(lastPositionBest.position));
    clock_t end_time = clock();

    // Tính thời gian thực thi (theo giây)
    double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // In ra thời gian thực thi
    printf("Thời gian thực thi: %f giây\n", execution_time);

    return 0;
}
