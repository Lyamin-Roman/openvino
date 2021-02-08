/*
// Copyright (c) 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include <gtest/gtest.h>
#include "api/memory.hpp"
#include <api/input_layout.hpp>
#include "api/ctc_greedy_decoder.hpp"
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include <api/mutable_data.hpp>
#include <api/data.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

TEST(ctc_gd_layer_tests, second_output) {
    static const int32_t N = 2, T = 8, C = 11;
    bool ctc_merge_repeated = false;
    int32_t blank_index = 5;
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ N, T, 1, C } });
    auto sequence_length = memory::allocate(engine, { data_types::i32, format::bfyx,{ N, 1, 1, 1 } });
    auto second_output = memory::allocate(engine, { data_types::i32, format::bfyx, { N, 1, 1, 1 } });
    topology topology;
    topology.add(input_layout("input", input.get_layout()));
    topology.add(data("sequence_length", sequence_length));
    topology.add(mutable_data("second_output", second_output));
    topology.add(ctc_greedy_decoder("ctc_greedy_decoder", "input", "sequence_length", blank_index,
                                    ctc_merge_repeated, tensor(format::bfyx, {N,T,1,1}), "second_output"));

    std::vector<float> input_vec = {
        0.0f, 1.0f, 8.0f, 5.0f, 5.0f, 2.0f, 0.0f, 7.0f,
        7.0f, 10.0f, 4.0f, 5.0f, 9.0f, 0.0f, 0.0f, 5.0f,
        7.0f, 0.0f, 4.0f, 0.0f, 4.0f, 7.0f, 6.0f, 10.0f,
        9.0f, 5.0f, 1.0f, 7.0f, 4.0f, 7.0f, 10.0f, 8.0f,
        2.0f, 0.0f, 8.0f, 3.0f, 6.0f, 8.0f, 10.0f, 4.0f,
        2.0f, 10.0f, 7.0f, 8.0f, 7.0f, 0.0f, 6.0f, 9.0f,
        2.0f, 4.0f, 8.0f, 5.0f, 2.0f, 3.0f, 3.0f, 1.0f,
        5.0f, 9.0f, 10.0f, 0.0f, 9.0f, 5.0f, 5.0f, 3.0f,
        10.0f, 5.0f, 2.0f, 0.0f, 10.0f, 0.0f, 5.0f, 4.0f,
        3.0f, 10.0f, 5.0f, 5.0f, 10.0f, 0.0f, 8.0f, 8.0f,
        9.0f, 1.0f, 0.0f, 7.0f, 9.0f, 6.0f, 8.0f, 7.0f,
        10.0f, 9.0f, 2.0f, 3.0f, 3.0f, 5.0f, 6.0f, 9.0f,
        4.0f, 9.0f, 2.0f, 4.0f, 5.0f, 5.0f, 3.0f, 1.0f,
        1.0f, 6.0f, 8.0f, 0.0f, 5.0f, 5.0f, 10.0f, 8.0f,
        6.0f, 9.0f, 6.0f, 9.0f, 1.0f, 2.0f, 7.0f, 1.0f,
        1.0f, 3.0f, 0.0f, 4.0f, 0.0f, 7.0f, 10.0f, 2.0f,
        1.0f, 3.0f, 9.0f, 7.0f, 1.0f, 7.0f, 4.0f, 4.0f,
        5.0f, 1.0f, 6.0f, 9.0f, 6.0f, 10.0f, 6.0f, 1.0f,
        10.0f, 4.0f, 1.0f, 6.0f, 2.0f, 5.0f, 5.0f, 10.0f,
        1.0f, 2.0f, 3.0f, 6.0f, 1.0f, 7.0f, 6.0f, 8.0f,
        2.0f, 5.0f, 4.0f, 2.0f, 0.0f, 9.0f, 4.0f, 1.0f,
        10.0f, 4.0f, 1.0f, 9.0f, 1.0f, 1.0f, 0.0f, 4.0f
    };
    set_values(input, input_vec);

    std::vector<int32_t> sequence_length_vec = { 3, 8 };
    set_values(sequence_length, sequence_length_vec);

    cldnn::build_options options;
    options.set_option(cldnn::build_option::optimize_data(false));

    network network(engine, topology, options);

    network.set_input_data("input", input);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "ctc_greedy_decoder");
    auto output = outputs.at("ctc_greedy_decoder").get_memory();
    auto output_ptr = output.pointer<float>();
    float out_buffer[N * T];
    for (uint32_t i = 0; i < N * T; i++) {
        out_buffer[i] = get_value<float>(output_ptr, i);
    }
    std::vector<float> expected_first_output = {
        9.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
        0.0f, 7.0f, 0.0f, 9.0f, 1.0f, 3.0f, -1.0f, -1.0f
    };
    for (int i = 0; i < N * T; i++) {
        EXPECT_EQ(out_buffer[i], expected_first_output[i]);
    }

    auto second_output_ptr = second_output.pointer<int32_t>();
    int32_t second_out_buffer[N];
    for (uint32_t i = 0; i < N; i++) {
        second_out_buffer[i] = get_value<int32_t>(second_output_ptr, i);
    }
    std::vector<int32_t> expected_second_output = {
        3, 6
    };
    for (int i = 0; i < N; i++) {
        EXPECT_EQ(second_out_buffer[i], expected_second_output[i]);
    }
}
