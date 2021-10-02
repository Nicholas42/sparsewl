#include <cstdio>
#include "src/AuxiliaryMethods.h"
#include "src/ColorRefinementKernel.h"
#include "src/ShortestPathKernel.h"
#include "src/GraphletKernel.h"
#include "src/GenerateTwo.h"
#include "src/GenerateThree.h"
#include "src/Graph.h"
#include <iostream>
#include <chrono>

using namespace std::chrono;
using namespace std;
using namespace GraphLibrary;
using namespace std;

//template<typename T>
//std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
//    if (!v.empty()) {
//        out << '[';
//        std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
//        out << "\b\b]";
//    }
//    return out;
//}

int main(int argc, char **argv) {
    const auto datasets = [&]() {
        if (argc <= 1) {
            std::cout << "Using default datasets\n";
            return vector<std::string>{"ENZYMES", "IMDB-BINARY", "IMDB-MULTI", "NCI1", "NCI109", "PTC_FM"};
        }
        return std::vector<std::string>(argv + 1, argv + argc);
    }();

    {
        for (auto &ds: datasets) {
            bool use_labels;
            {
                string kernel = "LWL2";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                use_labels = gdb[0].has_node_labels();
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                GenerateTwo::GenerateTwo wl(gdb);
                for (uint i = 0; i <= 5; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, false, "local", true, true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, false, "local", true, true);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes,
                                                   "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) +
                                                   ".gram");
                }
            }

            {

                string kernel = "LWLP2";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                GenerateTwo::GenerateTwo wl(gdb);
                for (uint i = 0; i <= 5; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, false, "localp", true, true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, false, "localp", true, true);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes,
                                                   "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) +
                                                   ".gram");
                }
            }
        }
    }

    {
        for (auto &ds: datasets) {
            bool use_labels;
            {
                string kernel = "WL2";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                use_labels = gdb[0].has_node_labels();
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                GenerateTwo::GenerateTwo wl(gdb);
                for (uint i = 0; i <= 5; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, false, "wl", true, true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, false, "wl", true, true);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes,
                                                   "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) +
                                                   ".gram");
                }
            }

            {
                string kernel = "DWL2";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                GenerateTwo::GenerateTwo wl(gdb);
                for (uint i = 0; i <= 5; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, false, "malkin", true, true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, false, "malkin", true, true);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes,
                                                   "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) +
                                                   ".gram");
                }
            }
        }
    }

    // k = 3.
    {
        for (auto &ds: datasets) {
            bool use_labels;
            {
                string kernel = "LWL3";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                use_labels = gdb[0].has_node_labels();
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                GenerateThree::GenerateThree wl(gdb);
                for (uint i = 0; i <= 5; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, "local", true, true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, "local", true, true);
                    }

                    AuxiliaryMethods::write_libsvm(
                        gm, classes, "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) + ".gram");
                }
            }

            {
                string kernel = "LWLP3";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                GenerateThree::GenerateThree wl(gdb);
                for (uint i = 0; i <= 5; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, "localp", true, true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, "localp", true, true);
                    }

                    AuxiliaryMethods::write_libsvm(
                        gm, classes, "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) + ".gram");
                }
            }
        }
    }

    {
        for (auto &ds: datasets) {
            bool use_labels;
            {
                string kernel = "WL3";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                use_labels = gdb[0].has_node_labels();
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                GenerateThree::GenerateThree wl(gdb);
                for (uint i = 0; i <= 5; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, "wl", true, true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, "wl", true, true);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes,
                                                   "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) +
                                                   ".gram");
                }
            }

            {
                string kernel = "DWL3";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                GenerateThree::GenerateThree wl(gdb);
                for (uint i = 0; i <= 5; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, "malkin", true, true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, "malkin", true, true);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes,
                                                   "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) +
                                                   ".gram");
                }
            }
        }
    }

    // Simple kernel baselines.
    {
        for (auto &ds: datasets) {
            bool use_labels;
            {
                string kernel = "WL1";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                use_labels = gdb[0].has_node_labels();
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                ColorRefinement::ColorRefinementKernel wl(gdb);
                for (uint i = 0; i <= 5; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, false, true, false);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, false, true, false);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes,
                                                   "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) +
                                                   ".gram");
                }
            }

            {
                string kernel = "WLOA";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                ColorRefinement::ColorRefinementKernel wl(gdb);
                for (uint i = 1; i <= 5; ++i) {
                    cout << ds + "__" + kernel + "_" + to_string(i) << endl;
                    GramMatrix gm;

                    if (i == 5) {
                        high_resolution_clock::time_point t1 = high_resolution_clock::now();
                        gm = wl.compute_gram_matrix(i, use_labels, false, true, true);
                        high_resolution_clock::time_point t2 = high_resolution_clock::now();
                        auto duration = duration_cast<seconds>(t2 - t1).count();
                        cout << duration << endl;
                    } else {
                        gm = wl.compute_gram_matrix(i, use_labels, false, true, true);
                    }

                    AuxiliaryMethods::write_libsvm(gm, classes,
                                                   "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(i) +
                                                   ".gram");
                }
            }

            {
                string kernel = "SP";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                ShortestPathKernel::ShortestPathKernel sp(gdb);

                cout << ds + "__" + kernel + "_" + to_string(0) << endl;
                GramMatrix gm;

                high_resolution_clock::time_point t1 = high_resolution_clock::now();
                gm = sp.compute_gram_matrix(use_labels, true);
                high_resolution_clock::time_point t2 = high_resolution_clock::now();
                auto duration = duration_cast<seconds>(t2 - t1).count();
                cout << duration << endl;

                AuxiliaryMethods::write_libsvm(gm, classes,
                                               "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(0) + ".gram");
            }

            {
                string kernel = "GR";
                GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(ds);
                gdb.erase(gdb.begin() + 0);
                vector<int> classes = AuxiliaryMethods::read_classes(ds);

                GraphletKernel::GraphletKernel sp(gdb);

                cout << ds + "__" + kernel + "_" + to_string(0) << endl;
                GramMatrix gm;

                high_resolution_clock::time_point t1 = high_resolution_clock::now();
                gm = sp.compute_gram_matrix(use_labels, false, true);
                high_resolution_clock::time_point t2 = high_resolution_clock::now();
                auto duration = duration_cast<seconds>(t2 - t1).count();
                cout << duration << endl;

                AuxiliaryMethods::write_libsvm(gm, classes,
                                               "./svm/GM/EXP/" + ds + "__" + kernel + "_" + to_string(0) + ".gram");
            }
        }
    }

    return 0;
}
