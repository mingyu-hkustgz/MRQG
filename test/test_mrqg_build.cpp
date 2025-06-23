#include <iostream>
#include <string>


#include "mrqg/mrqg.hpp"
#include "mrqg/mrqg_builder.hpp"
#include "utils/io.hpp"
#include "utils/stopw.hpp"

std::string dataset = "msmarc-small";
size_t degree = 32;
size_t flop_dim = 256;
auto data_file = "/DATA/" + dataset + "/" + dataset + "_proj.fvecs";
auto index_file = "./data/" + dataset + "/" + "symqg" + std::to_string(degree) + ".index";
int main() {
    using data_type = symqg::RowMatrix<float>;

    data_type data;

    symqg::load_vecs<float, data_type>(data_file.c_str(), data);
    StopW stopw;

    symqg::ResidualQuantizedGraph qg(data.rows(), degree, data.cols(), flop_dim);

    symqg::MRQGBuilder builder(qg, 400, data.data(), 9999);
    // 3 iters, refine at last iter
    builder.build(3);

    auto milisecs = stopw.get_elapsed_mili();

    std::cout << "Indexing time " << milisecs / 1000.F << " secs\n";

    qg.save_index(index_file.c_str());

    return 0;
}