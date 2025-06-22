//
// Created by bld on 25-6-22.
//

#include <iostream>
#include <string>
#include <getopt.h>
#include <ctime>

#include "mrqg/mrqg.hpp"
#include "mrqg/mrqg_builder.hpp"
#include "mrqg/mrqg_query.hpp"
#include "space/matrix.h"
#include "sys/resource.h"
#include "sys/time.h"

#include "utils/io.hpp"
#include "utils/stopw.hpp"

using namespace std;

long double rotation_time = 0;
int probe_base = 50;
char data_path[256] = "";

void GetCurTime(rusage *curTime) {
    int ret = getrusage(RUSAGE_SELF, curTime);
    if (ret != 0) {
        fprintf(stderr, "The running time info couldn't be collected successfully.\n");
        //FreeData( 2);
        exit(0);
    }
}

/*
* GetTime is used to get the 'float' format time from the start and end rusage structure.
*
* @Param timeStart, timeEnd indicate the two time points.
* @Param userTime, sysTime get back the time information.
*
* @Return void.
*/
void GetTime(struct rusage *timeStart, struct rusage *timeEnd, float *userTime, float *sysTime) {
    (*userTime) = ((float) (timeEnd->ru_utime.tv_sec - timeStart->ru_utime.tv_sec)) +
                  ((float) (timeEnd->ru_utime.tv_usec - timeStart->ru_utime.tv_usec)) * 1e-6;
    (*sysTime) = ((float) (timeEnd->ru_stime.tv_sec - timeStart->ru_stime.tv_sec)) +
                 ((float) (timeEnd->ru_stime.tv_usec - timeStart->ru_stime.tv_usec)) * 1e-6;
}


void test(const Matrix<float> &Q) {
    float sys_t, usr_t, usr_t_sum = 0, total_time = 0, search_time = 0;
    struct rusage run_start, run_end;

    // ========================================================================
    // Search Parameter
    // ========================================================================

    for (int nprobe = probe_base; nprobe <= probe_base * 20; nprobe += probe_base) {
        float total_time = 0;
        float total_ratio = 0;
        int correct = 0;
#ifdef COUNT_SCAN
        count_scan = 0;
        all_dist_count = 0;
#endif
        for (int i = 0; i < Q.n; i++) {
            GetCurTime(&run_start);

            GetCurTime(&run_end);
            GetTime(&run_start, &run_end, &usr_t, &sys_t);
            total_time += usr_t * 1e6;

            int tmp_correct = 0;
            while (KNNs.empty() == false) {
                int id = KNNs.top().second;
                KNNs.pop();
                for (int j = 0; j < k; j++)
                    if (id == G.data[i * G.d + j])tmp_correct++;
            }
            correct += tmp_correct;
        }
        float time_us_per_query = total_time / Q.n;
        float recall = 1.0f * correct / (Q.n * k);

        cout << recall * 100.0 << " " << 1e6 / (time_us_per_query) << endl;
    }
}

int main(int argc, char *argv[]) {

    const struct option longopts[] = {
            // General Parameter
            {"help",        no_argument,       0, 'h'},

            // Query Parameter
            {"K",           required_argument, 0, 'k'},

            // Indexing Path
            {"dataset",     required_argument, 0, 'd'},
            {"source",      required_argument, 0, 's'},
            {"result_path", required_argument, 0, 'r'},
    };

    int ind, bit;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    char dataset[256] = "";
    char source[256] = "";
    char result_path[256] = "";
    int subk = 0;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:r:k:s:b:", longopts, &ind);
        switch (iarg) {
            case 'k':
                if (optarg)subk = atoi(optarg);
                break;
            case 's':
                if (optarg)strcpy(source, optarg);
                break;
            case 'r':
                if (optarg)strcpy(result_path, optarg);
                break;
            case 'd':
                if (optarg)strcpy(dataset, optarg);
                break;
            case 'b':
                if (optarg) bit = atoi(optarg);
                break;
        }
    }

    // ================================================================================================================================
    // Data Files
    char query_path[256] = "";
    sprintf(query_path, "%s%s_query.fvecs", source, dataset);
    Matrix<float> Q(query_path);

    char mean_path[256] = "";
    sprintf(mean_path, "%s%s_mean.fvecs", source, dataset);
    Matrix<float> M(mean_path);

    sprintf(data_path, "%s%s_proj.fvecs", source, dataset);

    char groundtruth_path[256] = "";
    sprintf(groundtruth_path, "%s%s_groundtruth.ivecs", source, dataset);
    Matrix<unsigned> G(groundtruth_path);

    char pca_matrix_path[256] = "";
    sprintf(pca_matrix_path, "%s%s_pca.fvecs", source, dataset);
    Matrix<float> PCA(pca_matrix_path);

    char result_file_view[256] = "";
    std::cerr << result_file_view << std::endl;
    std::cerr << "Loading Succeed!" << std::endl;
    // ================================================================================================================================

    freopen(result_file_view, "a", stdout);
    float sys_t, usr_t, usr_t_sum = 0, total_time = 0, search_time = 0;
    struct rusage run_start, run_end;
    GetCurTime(&run_start);
    std::cerr << "begin Matrix Operation" << std::endl;
    Matrix<float> PCAQ(Q.n, Q.d, Q);
    PCAQ = mul(PCAQ, PCA);
    PCAQ = cen(PCAQ, M);
    GetCurTime(&run_end);
    GetTime(&run_start, &run_end, &usr_t, &sys_t);
    rotation_time = usr_t * 1e6 / Q.n;
    std::string str_data(dataset);
    std::cerr << "dataset:: " << str_data << std::endl;


    return 0;
}
