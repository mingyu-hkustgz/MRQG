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

std::string dataset = "msmarc-small";
size_t degree = 32, k=10;
size_t flop_dim = 256;
auto data_file = "/DATA/" + dataset + "/" + dataset + "_proj.fvecs";
auto query_file = "/DATA/" + dataset + "/" + dataset + "_query_proj.fvecs";
auto index_file = "./data/" + dataset + "/" + "symqg" + std::to_string(degree) + ".index";
long double rotation_time = 0;
int probe_base = 50;

int main(int argc, char *argv[]) {
    char groundtruth_path[256] = "/DATA/msmarc-small/msmarc-small_groundtruth.ivecs";
    using data_type = symqg::RowMatrix<float>;
    data_type data, query;
    symqg::load_vecs<float, data_type>(data_file.c_str(), data);
    symqg::load_vecs<float, data_type>(query_file.c_str(), query);
    StopW stopw;
    symqg::ResidualQuantizedGraph qg(data.rows(), degree, data.cols(), flop_dim);
    qg.load_index(index_file.c_str());
    Matrix<unsigned> G(groundtruth_path);
    std::cerr << "Loading Succeed!" << std::endl;
    // ================================================================================================================================

    float sys_t, usr_t, usr_t_sum = 0, total_time = 0, search_time = 0;
    struct rusage run_start, run_end;
    std::vector<uint32_t> KNNs(k * 1000);
    uint32_t  iter = 3;
    while(iter--) {
        for (int nprobe = probe_base; nprobe <= probe_base * 20; nprobe += probe_base) {
            float total_time = 0;
            float total_ratio = 0;
            int correct = 0;
            for (int i = 0; i < 1000; i++) {
                GetCurTime(&run_start);
                qg.set_ef(nprobe);
                qg.search(query.data() + i * data.cols(), k, KNNs.data() + i * k);
                GetCurTime(&run_end);
                GetTime(&run_start, &run_end, &usr_t, &sys_t);
                total_time += usr_t * 1e6;
            }
            for (int i = 0; i < KNNs.size();i++) {
                auto id = KNNs[i];
                for (int j = 0; j < k; j++)
                    if (id == G.data[i/k * G.d + j]) correct++;
            }
            float time_us_per_query = total_time / 1000;
            float recall = 1.0f * correct / (1000 * k);

            cout << recall * 100.0 << " " << (uint32_t) (1e6 / (time_us_per_query))<<" "<<nprobe << endl;
        }
    }
    return 0;
}
