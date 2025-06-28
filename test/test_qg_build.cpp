#include <iostream>
#include <string>

#include "qg/qg.hpp"
#include "qg/qg_builder.hpp"
#include "utils/io.hpp"
#include "utils/stopw.hpp"


int main(int argc, char *argv[]) {
    std::string dataset(argv[1]);
    size_t degree = 64;
    auto data_file = "/DATA/" + dataset + "/" + dataset + "_proj.fvecs";
    auto index_file = "./data/" + dataset + "/" + "symqg" + std::to_string(degree) + ".index";

    using data_type = symqg::RowMatrix<float>;

    data_type data;

    symqg::load_vecs<float, data_type>(data_file.c_str(), data);

    StopW stopw;

    symqg::QuantizedGraph qg(data.rows(), degree, data.cols());

    symqg::QGBuilder builder(qg, 400, data.data(), 9999);

    // 3 iters, refine at last iter
    builder.build(3);

    auto milisecs = stopw.get_elapsed_mili();

    std::cout << "Indexing time " << milisecs / 1000.F << " secs\n";

    qg.save_index(index_file.c_str());

    std::cout << "Indexing time " << milisecs / 1000.F << " secs\n";
    std::string time_path = "./results/index-time/" + dataset + ".log";
    std::ofstream fout(time_path, std::ios::app);
    fout<<"Index QG degree: "<<degree<<" time: "<<milisecs / 1000.F<<"(s)"<<std::endl;

    return 0;
}